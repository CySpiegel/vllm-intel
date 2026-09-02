# vllm-xpu-b70 — vLLM tuned for dual Intel Arc B-series (BMG), unofficial

Serving stack for 2x Intel Arc B70 (Battlemage, 32 GB each), built from
[CySpiegel/vllm-intel](https://github.com/CySpiegel/vllm-intel) and
[CySpiegel/vllm-xpu-kernels](https://github.com/CySpiegel/vllm-xpu-kernels).
Not an official vLLM or Intel image.

Image: `cyspiegel/vllm-xpu-b70` — tags `latest` == `2026-09-02-680f1e9` (vLLM fork
main 680f1e9a2c: same vLLM code as 47711e60e2 = upstream main ca90b9e7d 2026-08-27 + our
patches, including the two fixes also submitted upstream as #53996 and #53997; kernels wheel
vllm-xpu-kernels 0.1.14.dev10+g3b2c7e6; entrypoint now defaults prefix caching ON and the
image runs unprivileged). Previous tags `2026-08-27-47711e6` and `2026-08-27-a1c981a` stay available. On 2x Arc Pro B70, Qwen3.8-27B (INT4 + MTP)
runs at 57 tok/s single-stream / 203 tok/s batched inside this image vs 16 / 77 on the
stock `intel/vllm:0.21.0` image with the same INT4 weights (3.5x / 2.6x); Gemma 4 31B
INT4 at 33 / 148 vs 11 / ~30 (3x / ~5x). The container is within 2-5% of bare metal on the
same branch at TP=2 and within 1% on one card. Everything below was measured on this box; commands are included.

## Why this image exists

- Stock intel/vllm 0.21 runs these models eagerly under TP=2: its container log says "XPU Graph doesn't support capture communication ops,
  disabling cudagraph_mode" — every kernel of a 60-layer model launched from Python per
  token. That is the 3-5x.
- Bare `pip` vLLM on Battlemage needs a bring-up recipe (libze loader symlink,
  FI_PROVIDER, Triton arch env, ocloc, driver >= 26.18) that took days to find; the
  image bakes it (compute-runtime 26.27 + IGC 2.38).
- Speculative decoding (MTP) on Qwen3.5/3.8-class hybrid GDN models crashed under
  concurrency on XPU; fixed at kernel level here (see fixes) and validated under load
  (HumanEval + concurrent GSM8K).
- INT4 AutoRound checkpoints only reach speed on the GPTQ kernel path; the image
  auto-derives that variant (and handles Gemma 4's missing v_proj) so any AutoRound
  export just works.
- Every optimization was A/B benchmarked and kept only if it measured faster (ledger
  of ~20 entries).

## What was fixed or added

### vLLM (this fork; upstream PRs where filed)

- XPU graph capture kept working under TP=2 (stock compose env `CCL_ENABLE_SYCL_KERNELS=0`
  breaks it; image drops it). Eager -> graphs: 72 -> 31 ms/token on FP8 (ledger A1).
- FLASH_ATTN backend on XPU instead of TRITON_ATTN (A4).
- Fused QK-norm + RoPE + gate Triton kernel enabled on XPU
  (upstream PR [vllm-project/vllm#53989](https://github.com/vllm-project/vllm/pull/53989)).
- `getMemoryInfo` reports free=0 on Arc B-series -> vLLM cannot start; fallback to
  torch.xpu.mem_get_info
  (upstream PR [#53990](https://github.com/vllm-project/vllm/pull/53990)).
- Flag-gated Triton one-shot all-reduce for TP=2 (`VLLM_XPU_TRITON_ALLREDUCE=1`,
  default off; ~0.4 ms/step).
- Eagle/MTP spec-prep hardened against index corruption from concurrent first-compile
  (per-rank Triton caches, `VLLM_XPU_SYNC_AFTER_SPEC_PREP`).
- GPTQ loader: Gemma 4 global-attention layers have no v_proj -> vLLM's fused-shard
  check rejected the model (upstream issue #53992; fix in this image and upstream PR
  [#53996](https://github.com/vllm-project/vllm/pull/53996)); the image converter also
  emits per-block module hints.
- V2 model runner failed under XPU graphs (grammar-bitmask cross-stream wait inside SYCL
  graph recording; upstream issue #53993; fix in this image and upstream PR
  [#53997](https://github.com/vllm-project/vllm/pull/53997)). The image still defaults
  `VLLM_USE_V2_MODEL_RUNNER=0` because V1 is the benchmarked runner.
- Synced to upstream main ca90b9e7d (2026-08-27) with no measured regression (ledger R1).

### kernels (vllm-xpu-kernels)

- Fused gdn_attention: mixed speculative/non-speculative batches in one invocation (was a
  hard RuntimeError -> MTP unusable under concurrency; upstream issue #510 area).
- XE2 delta-rule epilogue wrote out of bounds with non-contiguous token_indx (silent
  corruption of neighbouring rows) — root-caused, fixed at the interface, bitwise
  differential test added
  (upstream PR [vllm-project/vllm-xpu-kernels#552](https://github.com/vllm-project/vllm-xpu-kernels/pull/552)).
- Spec conv-state written from registers instead of an epilogue roll (bit-identical,
  -1..-11% kernel time; upstream PR #551).
- Reconciled with upstream #544 conv-state layout; full GDN test matrix 1303/0/128.

### image/runtime

- compute-runtime 26.27 + IGC 2.38 in-image: fixes the torch.nonzero element-drop bug of
  the Ubuntu stock 26.05 driver (torch-xpu-ops #4396) regardless of host driver.
- Runs unprivileged: no `--privileged`, no `--cap-add`. Docker's default seccomp profile
  blocks oneCCL's pidfd IPC exchange (`pidfd_getfd` needs CAP_SYS_PTRACE), so oneCCL falls
  back to drmfd, which only needs `/dev/dri/by-path` bind-mounted (read-only). Validated:
  gate PASS, greedy 13.02 ms TPOT, batched 196.9 t/s — identical to the privileged runs.
- Presets: int4-mtp (default), int4, fp8, custom; AutoRound->GPTQ auto-conversion; parsers
  per model.
- FP8 preset: online per-tensor FP8 is the slowest single-stream option (~30 ms/token,
  container and bare metal alike); INT4 on the GPTQ kernel path is the fast one.

## Benchmarks

Hardware: 2x Intel Arc Pro B70 32 GB (Battlemage G31), host Ubuntu 26.04, driver 26.05
on host / 26.27 in-image; client `vllm bench serve` from the host against :8000. Legs:
greedy single (`--random-input-len 1024 --random-output-len 128 --num-prompts 8
--max-concurrency 1 --temperature 0 --save-detailed`), sampled single (same,
model-default temperature), batched (`--random-output-len 256 --num-prompts 64
--max-concurrency 32`); a 4-prompt correctness gate before each run. TPOT = median time
per output token, "per-step" = median inter-token latency of the engine (for MTP,
TPOT = step / (1 + acceptance)). All servers: `--max-num-seqs 8 --max-num-batched-tokens
8192 --gpu-memory-utilization 0.90`, prefix caching off. TP=2 Qwen runs at its full
native 262,144 context; TP=2 Gemma 4 at 65,536 (its native 262,144 does not fit two
32 GB cards at INT4 — the KV cache holds ~109k tokens after weights — so the largest
power of two that fits is used; the same rule sets every single-card context, printed in
each table). Stock image = `intel/vllm:0.21.0-ubuntu24.04-20260625` with Intel's compose
env; its Qwen number uses the SAME INT4 weights (`--dtype float16`).

![2x B70 results](https://raw.githubusercontent.com/CySpiegel/vllm-intel/main/docker/b70/assets/bench-tp2.png)

### Qwen3.8-27B — 2x Arc B70 (TP=2, context 262,144)

| Preset | Leg | Stock intel/vllm 0.21 | This image | Bare metal (same branch) |
| --- | --- | --- | --- | --- |
| int4-mtp | greedy single | n/a: stock has no working MTP; INT4 no-MTP 61.4 ms / 16.3 tok/s | 13.01 / 24.62 / 57.5 (acc ~0.89, TTFT 569) | 12.15 / 23.37 / 61.1 (TTFT 530) |
| int4-mtp | sampled single (T=1.0) | — | 16.09 / 25.74 / 48.8 | 14.45 / 24.48 / 51.7 |
| int4-mtp | batched conc-32 | 76.9 tok/s (INT4) | 202.9 tok/s (TPOT 33.3) | 209.1 (32.9) |
| int4 (no MTP) | greedy | 61.4 ms / 16.3 tok/s (fp16) | 19.01 / 43.3 (TTFT 546) | 19.3 ms / 51.9 tok/s (ledger A10) |
| int4 (no MTP) | batched | 76.9 tok/s | 201.4 tok/s (TPOT 27.2) | 206 tok/s (ledger A10) |
| fp8 (online, BF16 checkpoint) | greedy | 70.9 ms / 14.1 tok/s (TTFT 1204) | 29.68 / 29.6 (TTFT 562) | 29.50 / 30.5 (TTFT 448) |
| fp8 | batched | 65.7 tok/s | 164.3 tok/s | 166.7 tok/s |

Notes: MTP acceptance noise — the sampled-T=1.0 TPOT swings ±2 ms run to run with the
draft acceptance; per-step is the stable metric. Quality: INT4 vs FP8 HumanEval 93.9 vs
92.7, HumanEval+ 89.6 vs 90.9 (n=164, statistically tied).

### Gemma 4 31B (Intel/gemma-4-31B-it-int4-AutoRound) — 2x Arc B70 (TP=2, context 65,536)

| Metric | Stock intel/vllm 0.21 | This image | Bare metal (same branch) |
| --- | --- | --- | --- |
| greedy single | 89.4 ms / 11.0 tok/s (TTFT 300) | 26.15 / 33.1 (TTFT 551) | 25.39 / 34.1 (533) |
| sampled single | 89.7 / 10.9 | 27.04 / 32.6 | 26.43 / 33.2 |
| batched conc-32 | ~30 tok/s aggregate (run stopped at 42/64 after 6 min) | 148.0 (TPOT 40.9) | 150.4 (40.1) |

Note: no MTP drafter exists for Gemma 4 (no public checkpoint) -> 1 token/step; why it is
slower than Qwen: dense 60 softmax-attention layers x 16 KV heads vs Qwen's 48
linear-attention + 16 attention layers x 4 KV heads (per-step 25.4 vs 23.4 ms; the rest
of Qwen's lead is MTP).

### Single card (TP=1, GPU 0) — same models, same three stacks

![single-card results](https://raw.githubusercontent.com/CySpiegel/vllm-intel/main/docker/b70/assets/bench-tp1.png)

One Arc Pro B70 (32 GB) gets you the numbers below; note that on a single card the stock
image can capture graphs (its TP=2 problem does not apply), so the gap is smaller than on
two cards: the single-card edge is MTP + the kernel fixes. Context per row is the largest
power of two that fits at 0.90 util (vLLM's own estimate); FP8 Qwen does not fit at all
(weights 27.55 GiB of 30.3 GiB).

| Model | Stack | Context | Greedy TPOT / per-step / tok/s | Sampled tok/s | Batched tok/s | TTFT |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B INT4 (fp16, no MTP) | stock intel/vllm 0.21 | 131,072 | 33.9 / 33.9 / 22.4 | 24.1 | 107.5 | 1414 ms (first run) |
| Qwen3.8-27B INT4 + MTP | this image | 65,536 | 20.5 / 38.8 / 38.5 | 35.4 | 145.5 | 686 |
| Qwen3.8-27B INT4 + MTP | bare metal | 65,536 | 20.6 / 38.9 / 38.8 | 34.9 | 147.7 | 655 |
| Qwen3.8-27B INT4 (no MTP) | bare metal | 65,536 | 31.2 / 31.2 / 27.9 | 27.5 | 135.9 | 620 |
| Gemma 4 31B INT4 | stock intel/vllm 0.21 | 16,384 | 83.4 / 83.4 / 8.9 | 11.6 | 21.2 (13 min run) | 3724 first-run / 358 |
| Gemma 4 31B INT4 | this image | 4,096 | 41.3 / 41.2 / 21.6 | 21.2 | 63.9 | 695 |
| Gemma 4 31B INT4 | bare metal | 4,096 | 41.3 / 41.2 / 21.6 | 21.2 | 70.1 | 677 |
| Gemma 4 31B INT4, `--max-num-batched-tokens 2048` | bare metal | 8,192 | 41.5 / 41.3 / 21.6 | 21.2 | 69.8 | 677 |
| Qwen3.8-27B FP8 (online) | any | does not fit on one card (weights 27.55 GiB) | — | — | — | — |

Notes:

- Why the stock image fits more context on one card: no MTP drafter weights and no graph
  memory to reserve.
- Gemma's 4k cap at the default `--max-num-batched-tokens 8192` is the profile run's logits
  activation (262k vocab x 8192 tokens on one card), not the KV cache: at 2048 the same card
  fits 8,192 context with identical speed. Use that setting for Gemma on one card.
- Container == bare within 1% at TP=1 (no tensor-parallel all-reduce path).

## Reproduce

Docker run (from "Run" section below):

```bash
docker run --rm -it --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
  --ipc host --shm-size 16g -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/models:/models \
  -e MODEL=CySpiegel/Qwen3.8-27B-Int4-AutoRound \
  cyspiegel/vllm-xpu-b70:latest
```

Benchmark client (run on the host against the container; `--model` is only used for the
tokenizer, so point it at the checkpoint directory or HF id; `--served-model-name` must match
`SERVED_NAME`):

```bash
# greedy single-stream
vllm bench serve --base-url http://localhost:8000 --model <checkpoint-dir-or-hf-id> \
  --served-model-name Qwen3.8-27B-Int4 --dataset-name random \
  --random-input-len 1024 --random-output-len 128 --num-prompts 8 --max-concurrency 1 \
  --temperature 0 --save-result --save-detailed
# sampled single-stream (model-default temperature): same without --temperature 0
# batched, 32 concurrent
vllm bench serve --base-url http://localhost:8000 --model <checkpoint-dir-or-hf-id> \
  --served-model-name Qwen3.8-27B-Int4 --dataset-name random \
  --random-input-len 1024 --random-output-len 256 --num-prompts 64 --max-concurrency 32 \
  --save-result --save-detailed
```

Before each run, a 4-prompt coherence check verifies the model is responding sanely.
Results dirs and the full ledger (~20 A/B entries) live in the maintainer's local bench
ledger; the numbers in this README are copied from it verbatim.

## Tracking

Performance numbers and fixes are tracked at:

- [GitHub releases](https://github.com/CySpiegel/vllm-intel/releases) of
  CySpiegel/vllm-intel (one per image tag, with the full benchmark table).
- Pinned [Performance tracking issue #7](https://github.com/CySpiegel/vllm-intel/issues/7)
  (one comment per image tag).
- Upstream PRs and issues:
    - vllm: [#53989](https://github.com/vllm-project/vllm/pull/53989),
    [#53990](https://github.com/vllm-project/vllm/pull/53990),
    [#53996](https://github.com/vllm-project/vllm/pull/53996),
    [#53997](https://github.com/vllm-project/vllm/pull/53997)
    (issues [#53992](https://github.com/vllm-project/vllm/issues/53992),
    [#53993](https://github.com/vllm-project/vllm/issues/53993))
    - vllm-xpu-kernels: [#551](https://github.com/vllm-project/vllm-xpu-kernels/pull/551),
    [#552](https://github.com/vllm-project/vllm-xpu-kernels/pull/552)
    - Fork: [vllm-intel #3](https://github.com/CySpiegel/vllm-intel/issues/3),
    [#4](https://github.com/CySpiegel/vllm-intel/issues/4),
    [#5](https://github.com/CySpiegel/vllm-intel/issues/5),
    [vllm-xpu-kernels #4](https://github.com/CySpiegel/vllm-xpu-kernels/issues/4).

## Bare metal install — the same stack without Docker

Tested on Ubuntu 26.04, 2x Arc Pro B70, Python 3.12, torch 2.13.0+xpu, triton-xpu 3.7.2,
oneccl 2022.0.0.

1. **Driver + tools (apt):**
   ```bash
   sudo apt install libze1 libze-intel-gpu1 intel-opencl-icd libze-dev \
     intel-ocloc clinfo
   ```

   **Driver note:** Ubuntu 26.04 ships compute-runtime 26.05.37020, which has a
   `torch.nonzero` element-drop bug (intel/torch-xpu-ops#4396, fixed in build >= 38646 /
   release >= 26.18). The image ships 26.27 for that reason; on bare metal upgrade via
   Intel's [dgpu-docs](https://dgpu-docs.intel.com/) repo or accept the caveat. User
   must be in groups `render` and `video` (re-login).

2. **Python env (never system pip):**
   ```bash
   uv venv --python 3.12 && source .venv/bin/activate
   ```

3. **vLLM from this fork:**
   ```bash
   git clone https://github.com/CySpiegel/vllm-intel && cd vllm-intel
   uv pip install -r requirements/xpu.txt
   VLLM_TARGET_DEVICE=xpu uv pip install --no-build-isolation -e . -v
   ```
   (Pure Python on XPU: no CUDA/SYCL compile step; all device kernels come from
   vllm-xpu-kernels.)

4. **Custom kernels** (one of):

   a. Prebuilt wheel from the GitHub release of the matching image tag
      (<https://github.com/CySpiegel/vllm-intel/releases> — asset
      `vllm_xpu_kernels-0.1.14.dev10+g3b2c7e6-cp312-cp312-linux_x86_64.whl`, cp312, x86_64):
      ```bash
      uv pip install --no-deps --force-reinstall ./vllm_xpu_kernels-*.whl
      ```

   b. Build from source:
      ```bash
      git clone https://github.com/CySpiegel/vllm-xpu-kernels && cd vllm-xpu-kernels
      MAX_JOBS=8 uv pip install --no-build-isolation -e . -v
      ```
      ~3 h on 8 cores; the flash-attn template instantiations need 8-24 GB RSS per job,
      so keep MAX_JOBS <= 8 on a 128 GB box.

5. **oneCCL dlopen fix** (pip oneccl wheel loads unversioned libze_loader.so):
   ```bash
   ln -sf /usr/lib/x86_64-linux-gnu/libze_loader.so.1 \
     .venv/lib/python3.12/site-packages/torch/lib/libze_loader.so
   ```
   (Or `apt install libze-dev`, which provides it; the symlink is wiped if torch is
   reinstalled.)

6. **Environment for every multi-GPU run** (put in a script):
   ```bash
   export FI_PROVIDER_PATH=$PWD/.venv/lib FI_PROVIDER=tcp LD_LIBRARY_PATH=$PWD/.venv/lib
   export TRITON_INTEL_DEVICE_ARCH=bmg VLLM_XPU_ENABLE_XPU_GRAPH=1 HF_HUB_OFFLINE=1
   ```

   Why each:
   - libfabric providers live in the venv (oneccl wheel)
   - psm3 auto-pick -> ENOMEM and shm fails at the default memlock, so tcp
   - Triton cannot autodetect the arch in spawned workers
   - graphs are the biggest single win
   - offline avoids Hub round-trips
   - no `CCL_ZE_IPC_EXCHANGE` and no elevated rights: oneCCL's pidfd IPC exchange works
     between same-user worker processes at Ubuntu's default `kernel.yama.ptrace_scope=1`
     (oneCCL sets `PR_SET_PTRACER` itself), so no sudo / `CAP_SYS_PTRACE`; if pidfd is ever
     blocked it falls back to drmfd via `/dev/dri/by-path`. Membership in `render` and
     `video` is the only requirement.

7. **INT4 weights:** Any AutoRound export (`auto_round:auto_gptq`, sym) must be served
   through the GPTQ kernel path. Derive the variant once:
   ```bash
   python docker/b70/make_gptq_variant.py <path-to-hf-snapshot> <dest-dir>
   ```
   (Hardlinks the shards, rewrites config.json; handles Gemma 4's missing v_proj.)

   Checkpoints:
   - Qwen: [CySpiegel/Qwen3.8-27B-Int4-AutoRound](https://huggingface.co/CySpiegel/Qwen3.8-27B-Int4-AutoRound)
   - Gemma: [Intel/gemma-4-31B-it-int4-AutoRound](https://huggingface.co/Intel/gemma-4-31B-it-int4-AutoRound)

8. **Serve — the exact preset commands:**

   int4-mtp:
   ```bash
   vllm serve <qwen-gptq-variant> --served-model-name Qwen3.8-27B-Int4 \
     --tensor-parallel-size 2 --attention-backend FLASH_ATTN \
     --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192 \
     --gpu-memory-utilization 0.90 --enable-prefix-caching \
     --language-model-only --reasoning-parser qwen3 \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
     --host 0.0.0.0 --port 8000
   ```

   int4 (same without --speculative-config):
   ```bash
   vllm serve <qwen-gptq-variant> --served-model-name Qwen3.8-27B-Int4 \
     --tensor-parallel-size 2 --attention-backend FLASH_ATTN \
     --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192 \
     --gpu-memory-utilization 0.90 --enable-prefix-caching \
     --language-model-only --reasoning-parser qwen3 \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --host 0.0.0.0 --port 8000
   ```

   fp8:
   ```bash
   VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 vllm serve Qwen/Qwen3.8-27B \
     --quantization fp8 --served-model-name Qwen3.8-27B --tensor-parallel-size 2 \
     --attention-backend FLASH_ATTN --max-model-len 262144 --max-num-seqs 8 \
     --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
     --enable-prefix-caching --language-model-only --reasoning-parser qwen3 \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --host 0.0.0.0 --port 8000
   ```

   Gemma 4:
   ```bash
   VLLM_USE_V2_MODEL_RUNNER=0 vllm serve <gemma-gptq-variant> \
     --served-model-name gemma-4-31b-int4 --tensor-parallel-size 2 \
     --attention-backend FLASH_ATTN --max-model-len 65536 --max-num-seqs 8 \
     --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
     --no-enable-prefix-caching --language-model-only --reasoning-parser gemma4 \
     --enable-auto-tool-choice --tool-call-parser gemma4 --host 0.0.0.0 --port 8000
   ```
   (The fork carries the V2-runner fix, vllm-project/vllm#53997; V1 stays the benchmarked
   default.)

   Single card: add `ZE_AFFINITY_MASK=0`, `--tensor-parallel-size 1`, and the context
   caps from the single-card table (Qwen INT4 65536; Gemma 4 8192 with
   `--max-num-batched-tokens 2048`).

9. **Sanity check:**
   ```bash
   curl localhost:8000/v1/models
   ```
   Expected greedy numbers are in the Benchmarks section (bare-metal column) — if you
   are far below them, the usual cause is graphs not capturing (check the log for
   "XPU Graph" warnings) or the tcp/libfabric envs missing.

## Model weights

| Preset | Checkpoint | Notes |
| --- | --- | --- |
| `int4-mtp`, `int4` | [CySpiegel/Qwen3.8-27B-Int4-AutoRound](https://huggingface.co/CySpiegel/Qwen3.8-27B-Int4-AutoRound) | my AutoRound INT4 export; recipe, evals, and the config-only GPTQ-path rewrite are documented on the model card |
| `fp8` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | BF16 original, quantized to FP8 at load |

Set `MODEL` to either repo id (downloaded into the mounted HF cache on first start) or to
a local directory under `/models`.

## Run

Compose (recommended):

```yaml
services:
  vllm-b70:
    image: cyspiegel/vllm-xpu-b70:latest
    ipc: host
    shm_size: "16g"
    devices: ["/dev/dri:/dev/dri"]
    ports: ["8000:8000"]
    volumes:
      - /dev/dri/by-path:/dev/dri/by-path:ro   # oneCCL drmfd IPC exchange (no --privileged needed)
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
docker run --rm -it --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
  --ipc host --shm-size 16g -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v /path/to/models:/models \
  -e MODEL=CySpiegel/Qwen3.8-27B-Int4-AutoRound \
  cyspiegel/vllm-xpu-b70:latest
```

On first start with the
[CySpiegel/Qwen3.8-27B-Int4-AutoRound](https://huggingface.co/CySpiegel/Qwen3.8-27B-Int4-AutoRound)
checkpoint (or any AutoRound export) the entrypoint derives the gptq-config
variant (config rewrite + links, no tensor changes) into `/models` so the fast
XPUwNa16 GEMM path is used; set `AUTO_GPTQ_VARIANT=0` to skip.

## Presets and knobs

`PRESET=int4-mtp` (default) | `int4` | `fp8` | `custom` (`vllm serve` passthrough).
`MODEL`, `SERVED_NAME`, `PORT`, `HOST`, `TP`, `MAX_MODEL_LEN` (262144),
`MAX_NUM_SEQS` (8), `MAX_NUM_BATCHED_TOKENS` (8192), `GPU_MEMORY_UTILIZATION`
(0.90), `ENABLE_THINKING=0`, `PREFIX_CACHING=0` (default on), `FI_PROVIDER` (tcp|shm),
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
