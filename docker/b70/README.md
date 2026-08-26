# vllm-xpu-b70 — vLLM tuned for dual Intel Arc B-series (BMG), unofficial

Serving stack for 2x Intel Arc B70 (Battlemage, 32 GB each), built from
[CySpiegel/vllm-intel](https://github.com/CySpiegel/vllm-intel) and
[CySpiegel/vllm-xpu-kernels](https://github.com/CySpiegel/vllm-xpu-kernels).
Not an official vLLM or Intel image.

## What's inside

- vLLM XPU build of the fork (`main`): fused QK-norm+RoPE Triton kernel on XPU,
  flag-gated Triton one-shot all-reduce for TP=2 (`VLLM_XPU_TRITON_ALLREDUCE=1`),
  MTP speculative decoding on Qwen3.5/3.8-class GDN hybrids, startup kernel warmup
  and per-rank Triton caches (cold-start hardening).
- Custom `vllm-xpu-kernels` wheel: GDN mixed spec/non-spec batch fixes, sliding-window
  conv-state convention (#544), XE2 delta-epilogue OOB fix, spec conv-state
  register write-back.
- Intel compute-runtime 26.27 / IGC 2.38 in-image (fixes the `torch.nonzero`
  element-drop bug present in the Ubuntu 26.04 stock 26.05 driver).

## Measured — Qwen3.8-27B, TP=2, 2x Arc B70 (host-side numbers, same stack)

| Preset | Single-stream | Batched (conc 32) | vs stock intel/vllm 0.21 FP8 (14–16 tok/s) |
| --- | --- | --- | --- |
| `int4-mtp` | 57 tok/s greedy (13.1 ms/token), ~54 tok/s at T=1.0 | 199 tok/s | ~3.5x |
| `int4` | ~52 tok/s | ~206 tok/s | ~3.3x |
| `fp8` | ~30 tok/s in-container (33 bare metal) | 166 tok/s | ~2.1x |

`int4-mtp` and `fp8` numbers above were measured inside this image on 2x Arc B70
(`docker/b70/docker-compose.yaml`); `int4` is the bare-metal figure for the same stack.

INT4 quality is statistically tied with FP8 (HumanEval 93.9 vs 92.7, HumanEval+
89.6 vs 90.9, n=164). Weights: [CySpiegel/Qwen3.8-27B-Int4-AutoRound](https://huggingface.co/CySpiegel/Qwen3.8-27B-Int4-AutoRound).

## Run

Compose (recommended):

```yaml
services:
  vllm-b70:
    image: cyspiegel/vllm-xpu-b70:latest
    privileged: true      # oneCCL's pidfd IPC exchange (set in-image) needs ptrace rights
    ipc: host
    shm_size: "16g"
    devices: ["/dev/dri:/dev/dri"]
    ports: ["8000:8000"]
    volumes:
      - /path/to/models:/models
      - ~/.cache/huggingface:/root/.cache/huggingface
      - vllm-b70-cache:/root/.cache/vllm     # keeps torch.compile artifacts across restarts
      - vllm-b70-triton:/root/.triton
    environment:
      - PRESET=int4-mtp
      - MODEL=CySpiegel/Qwen3.8-27B-Int4-AutoRound
volumes:
  vllm-b70-cache:
  vllm-b70-triton:
```

The full file with comments is `docker/b70/docker-compose.yaml` in the repo.
The image carries the validated environment; **do not copy the `CCL_*` / `ZE_*`
block from the stock `intel/vllm` compose** — `CCL_ENABLE_SYCL_KERNELS=0` in
particular selects a oneCCL all-reduce that cannot be recorded into XPU graphs
and aborts graph capture on this stack.

One-liner:

```bash
docker run --rm -it --privileged --device /dev/dri --ipc host --shm-size 16g -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v /path/to/models:/models \
  -e MODEL=CySpiegel/Qwen3.8-27B-Int4-AutoRound \
  cyspiegel/vllm-xpu-b70:latest
```

On first start with an AutoRound HF checkpoint the entrypoint derives the
gptq-config variant (config rewrite + links, no tensor changes) into `/models`
so the fast XPUwNa16 GEMM path is used; set `AUTO_GPTQ_VARIANT=0` to skip.

## Presets and knobs

`PRESET=int4-mtp` (default) | `int4` | `fp8` | `custom` (`vllm serve` passthrough).
`MODEL`, `SERVED_NAME`, `PORT`, `HOST`, `TP`, `MAX_MODEL_LEN` (262144),
`MAX_NUM_SEQS` (8), `MAX_NUM_BATCHED_TOKENS` (8192), `GPU_MEMORY_UTILIZATION`
(0.90), `ENABLE_THINKING=0`, `PREFIX_CACHING=1`, `FI_PROVIDER` (tcp|shm),
`EXTRA_ARGS`, plus any positional `vllm serve` flags.
`TRITON_INTEL_DEVICE_ARCH=bmg` is set in-image (mandatory for spawn workers).

## Build from source

```bash
git clone https://github.com/CySpiegel/vllm-intel && cd vllm-intel
docker build -f docker/Dockerfile.xpu --target vllm-openai -t cyspiegel/vllm-xpu-b70:base .
# kernel wheel: build github.com/CySpiegel/vllm-xpu-kernels (main) and drop it in docker/b70/wheels/
mkdir -p docker/b70/wheels && cp /path/to/vllm_xpu_kernels-*.whl docker/b70/wheels/
docker build -f docker/b70/Dockerfile --build-arg KERNEL_WHEEL=<file>.whl -t cyspiegel/vllm-xpu-b70:latest .
```

## Support and issues

Bugs, questions, and feature requests for this image go to the GitHub issue tracker:
**<https://github.com/CySpiegel/vllm-intel/issues>** (vLLM side, image, presets, entrypoint).
Kernel-level problems (GDN / attention / GEMM on BMG):
<https://github.com/CySpiegel/vllm-xpu-kernels/issues>.
Please include the image tag (`docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' cyspiegel/vllm-xpu-b70:latest`),
your `docker run`/compose config, and the container log around the failure.

Source: <https://github.com/CySpiegel/vllm-intel> (this file lives at `docker/b70/README.md`).

## Licenses

vLLM and vllm-xpu-kernels: Apache-2.0. Qwen weights: Apache-2.0 (Alibaba/Qwen).
