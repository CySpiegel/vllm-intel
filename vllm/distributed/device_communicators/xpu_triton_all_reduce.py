# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused Triton one-shot all-reduce for XPU TP=2 over PCIe P2P.

Push-mode design for links WITHOUT P2P atomics (zeDeviceGetP2PProperties
flags=0x1 on 2x Arc B70): each rank streams its input into the peer's
symmetric-memory slot, stamps a per-block sequence flag in the peer's
mailbox, then spins on its OWN local mailbox. One kernel launch per call.

Hardware-validated design points (evidence: .claude/allreduce/diag.py runs,
2026-08-24; spec .claude/specs/xpu-custom-allreduce.md Stage 1b):
- Poll primitive: `tl.atomic_cas(flag, seq, seq, sem="acquire")`. Plain
  volatile loads are compiler-honored but NOT hardware-coherent against
  incoming P2P writes (spin never observes the peer's stamp); atomic_add
  polling works but is an RMW; CAS(seq,seq) is the pure-read form.
- Data slots are double-buffered by seq parity: with in-order queues the
  peer's next-call push can otherwise overwrite a slot mid-read.
- Eligibility REQUIRES numel % 1024 == 0. Ragged sizes produce partially
  vectorized P2P stores whose tail bytes linger in write-combine buffers
  and can be overtaken by the doorbell (observed as every-8th-element
  stale stripes); no Triton 3.7.2 construct flushes cross-PCIe WC (local
  fences don't, flush-reads only drain already-emitted TLPs). Engine
  decode shapes are batch*hidden(5120) — always 1024-aligned. Validated:
  20k randomized aligned + 200k back-to-back iters, zero errors; 27.0
  us/op at 10KB vs 42.6 oneCCL (eager, same-loop methodology).
- P2P READS cache after first touch on this platform — pull-mode designs
  do not work; do not "simplify" this kernel into one.
"""

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

BLOCK = 1024
MAX_NUMEL = 262144  # 512KB bf16 — decode shapes only; larger falls back
MAX_BLOCKS = (MAX_NUMEL + BLOCK - 1) // BLOCK


@triton.jit
def _one_shot_push_kernel(
    inp_ptr,
    out_ptr,
    local_slot_ptr,  # my slots (2x parity): peer pushes its data here
    peer_slot_ptr,  # peer's slots mapped into my VA: I push my data here
    local_flags_ptr,  # my mailbox: peer stamps seq here (I spin locally)
    peer_flags_ptr,  # peer's mailbox mapped into my VA: I stamp seq here
    fence_ptr,  # local scratch word for the release fence
    seq_ptr,  # device-side call counter (graph-safe: incremented by a
    # captured tensor op, never a host-baked constant)
    numel,
    SLOT_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    seq = tl.load(seq_ptr)
    parity = seq % 2
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(inp_ptr + offs, mask=mask)
    tl.store(peer_slot_ptr + parity * SLOT_STRIDE + offs, x, mask=mask)
    # Fire-and-forget non-relaxed atomic on local memory emits a GLOBAL
    # memfence in the XPU backend, ordering the data stores before the stamp.
    tl.atomic_xchg(fence_ptr, seq, sem="acq_rel", scope="sys")
    tl.store(peer_flags_ptr + pid, seq)
    # Clobber-free coherent poll: CAS(expected=seq, new=seq) can only write
    # seq-over-seq, so it can never destroy an incoming stamp.
    seen = tl.atomic_cas(local_flags_ptr + pid, seq, seq, sem="acquire")
    while seen < seq:
        seen = tl.atomic_cas(local_flags_ptr + pid, seq, seq, sem="acquire")
    y = tl.load(local_slot_ptr + parity * SLOT_STRIDE + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


class OneShotAllReduce:
    """TP=2 one-shot all-reduce over symmetric memory + Triton (XPU)."""

    def __init__(self, group: dist.ProcessGroup, device: torch.device):
        import torch.distributed._symmetric_memory as symm

        self.group = group
        self.device = device
        self.rank = group.rank()
        assert group.size() == 2, "one-shot path is TP=2 only"
        gname = group.group_name
        symm.enable_symm_mem_for_group(gname)
        self._slot = symm.empty(2 * MAX_NUMEL, dtype=torch.bfloat16, device=device)
        self._flags = symm.empty(MAX_BLOCKS, dtype=torch.int32, device=device)
        self._slot.zero_()
        self._flags.zero_()
        slot_hdl = symm.rendezvous(self._slot, group=gname)
        flag_hdl = symm.rendezvous(self._flags, group=gname)
        peer = 1 - self.rank
        self._peer_slot = slot_hdl.get_buffer(peer, self._slot.shape, self._slot.dtype)
        self._peer_flags = flag_hdl.get_buffer(
            peer, self._flags.shape, self._flags.dtype
        )
        self._fence = torch.zeros(1, dtype=torch.int32, device=device)
        self._seq_buf = torch.zeros(1, dtype=torch.int32, device=device)
        dist.barrier(group=group)

    def capture(self):
        # Graph-capture context (CustomAllreduce protocol). The kernel is
        # capture-safe as-is: fixed buffer addresses, device-side seq counter.
        from contextlib import nullcontext

        return nullcontext()

    def close(self):
        pass

    def should_custom_ar(self, t: torch.Tensor) -> bool:
        return (
            t.dtype == torch.bfloat16
            and t.is_contiguous()
            and 0 < t.numel() <= MAX_NUMEL
            and t.numel() % BLOCK == 0  # aligned regime only — see module doc
        )

    def all_reduce(self, t: torch.Tensor, out: torch.Tensor | None = None):
        if out is None:
            out = torch.empty_like(t)
        self._seq_buf += 1  # device op — replayed correctly inside graphs
        numel = t.numel()
        grid = (numel // BLOCK,)
        _one_shot_push_kernel[grid](
            t,
            out,
            self._slot,
            self._peer_slot,
            self._flags,
            self._peer_flags,
            self._fence,
            self._seq_buf,
            numel,
            SLOT_STRIDE=MAX_NUMEL,
            BLOCK_SIZE=BLOCK,
        )
        return out


def _worker(rank: int, world: int, iters_g1: int, iters_g2: int):
    import os
    import time

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29519"
    os.environ["ZE_AFFINITY_MASK"] = "0,1"
    os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")

    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", rank=rank, world_size=world)
    ar = OneShotAllReduce(dist.group.WORLD, torch.device(f"xpu:{rank}"))
    dev = f"xpu:{rank}"

    # G1: randomized correctness in the aligned regime, CPU-checked
    g = torch.Generator(device="cpu").manual_seed(1234)
    mism = 0
    for i in range(iters_g1):
        numel = BLOCK * int(torch.randint(1, MAX_BLOCKS + 1, (1,), generator=g).item())
        base = torch.randn(numel, generator=g, dtype=torch.float32)
        a = base.to(torch.bfloat16)
        b = (base + 0.5).to(torch.bfloat16)
        mine = (a if rank == 0 else b).to(dev)
        expect = a.to(dev) + b.to(dev)
        got = ar.all_reduce(mine)
        if not torch.equal(got, expect):
            # device compare cross-checked on CPU (XPU nonzero/odd-size quirks)
            mism += int((got.cpu() != expect.cpu()).sum() > 0)
    torch.xpu.synchronize()
    if rank == 0:
        print(f"[G1] {iters_g1} aligned iters, mismatches: {mism}", flush=True)

    # G2: ordering stress — back-to-back alternating-value calls
    t = torch.zeros(5120, dtype=torch.bfloat16, device=dev)
    out = torch.empty_like(t)
    bad = torch.zeros((), dtype=torch.int32, device=dev)
    for i in range(iters_g2):
        t.fill_(float(i % 7 + rank))
        ar.all_reduce(t, out)
        bad += (out != float(i % 7) * 2 + 1.0).any().to(torch.int32)
    torch.xpu.synchronize()
    if rank == 0:
        print(f"[G2] {iters_g2} stress iters, bad calls: {int(bad.item())}", flush=True)

    # G3: latency at 10KB, same methodology as the oneCCL baseline numbers
    t = torch.ones(5120, dtype=torch.bfloat16, device=dev)
    out = torch.empty_like(t)
    for _ in range(200):
        ar.all_reduce(t, out)
    torch.xpu.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    n = 2000
    for _ in range(n):
        ar.all_reduce(t, out)
    torch.xpu.synchronize()
    us = (time.perf_counter() - t0) / n * 1e6
    oc = torch.ones(5120, dtype=torch.bfloat16, device=dev)
    t0 = time.perf_counter()
    for _ in range(n):
        dist.all_reduce(oc)
    torch.xpu.synchronize()
    us_ccl = (time.perf_counter() - t0) / n * 1e6
    if rank == 0:
        print(f"[G3] 10KB one-shot: {us:.1f} us/op vs oneCCL {us_ccl:.1f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    mp.spawn(_worker, args=(2, 20000, 200000), nprocs=2, join=True)
