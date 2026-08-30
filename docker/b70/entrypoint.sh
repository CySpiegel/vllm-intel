#!/bin/bash
# Entrypoint for cyspiegel/vllm-xpu-b70.
#
# PRESET=int4-mtp (default) | int4 | fp8 | custom
#   int4-mtp : INT4 W4A16 AutoRound checkpoint + MTP speculative decoding (k=1)
#   int4     : same checkpoint, no speculative decoding
#   fp8      : BF16 Qwen/Qwen3.8-27B with online per-tensor FP8 quantization
#   custom   : exec `vllm serve "$@"` verbatim (no preset flags)
#
# MODEL may be a local directory (mount it under /models) or an HF repo id
# (downloaded into /root/.cache/huggingface on first start). For the INT4
# presets it defaults to /models/Qwen3.8-27B-Int4-AutoRound/gptq-variant when
# that directory exists, else to CySpiegel/Qwen3.8-27B-Int4-AutoRound.
#
# Knobs (all optional): MODEL, SERVED_NAME, PORT, HOST, TP, MAX_MODEL_LEN,
#   MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, GPU_MEMORY_UTILIZATION,
#   ENABLE_THINKING=0, PREFIX_CACHING=1, AUTO_GPTQ_VARIANT=0, GPTQ_VARIANT_DIR,
#   HF_HUB_OFFLINE (defaults to 1 once a local checkpoint is resolved),
#   FI_PROVIDER (default tcp; shm needs --privileged), EXTRA_ARGS.
# Positional arguments are appended to the vllm serve command line.
set -euo pipefail

PRESET=${PRESET:-int4-mtp}
export FI_PROVIDER=${FI_PROVIDER:-tcp}

if [ "$PRESET" = "custom" ]; then
  exec vllm serve "$@"
fi

DEFAULT_INT4_DIR=/models/Qwen3.8-27B-Int4-AutoRound/gptq-variant
DEFAULT_INT4_REPO=CySpiegel/Qwen3.8-27B-Int4-AutoRound

case "$PRESET" in
  int4-mtp|int4)
    if [ -z "${MODEL:-}" ]; then
      if [ -d "$DEFAULT_INT4_DIR" ]; then MODEL=$DEFAULT_INT4_DIR; else MODEL=$DEFAULT_INT4_REPO; fi
    fi
    SERVED_NAME=${SERVED_NAME:-Qwen3.8-27B-Int4}
    ;;
  fp8)
    MODEL=${MODEL:-Qwen/Qwen3.8-27B}
    SERVED_NAME=${SERVED_NAME:-Qwen3.8-27B}
    export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=${VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT:-1}
    ;;
  *)
    echo "entrypoint: unknown PRESET '$PRESET' (int4-mtp|int4|fp8|custom)" >&2
    exit 2
    ;;
esac

resolve_model() {
  # Local directory -> as-is. Path-shaped but missing -> clear error.
  # Otherwise an HF repo id -> snapshot_download (honours HF_HUB_OFFLINE if set).
  if [ -d "$MODEL" ]; then
    echo "$MODEL"
    return 0
  fi
  case "$MODEL" in
    /*|./*|../*)
      echo "entrypoint: MODEL directory not found: $MODEL" >&2
      return 1
      ;;
  esac
  python3 - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
PY
}

is_autoround_export() {
  python3 - "$1" <<'PY'
import json, sys
try:
    cfg = json.load(open(sys.argv[1] + "/config.json"))
except (OSError, ValueError) as e:
    print(f"entrypoint: cannot read {sys.argv[1]}/config.json: {e}", file=sys.stderr)
    sys.exit(2)
sys.exit(0 if cfg.get("quantization_config", {}).get("quant_method") == "auto-round" else 1)
PY
}

case "$PRESET" in
  int4-mtp|int4)
    MODEL_PATH=$(resolve_model)
    if [ "${AUTO_GPTQ_VARIANT:-1}" = "1" ]; then
      rc=0; is_autoround_export "$MODEL_PATH" || rc=$?
      [ "$rc" -eq 2 ] && exit 1
      if [ "$rc" -eq 0 ]; then
        # AutoRound exports serve through the INC/ARK path (slow batched); the
        # gptq-config variant of the same tensors takes the XPUwNa16 GEMM path
        # (+58% single-stream, +23% batched). Derive it once, idempotently.
        if [ -d "$MODEL" ]; then base=$(basename "$MODEL"); else base=${MODEL//\//--}; fi
        DEST=${GPTQ_VARIANT_DIR:-}
        if [ -z "$DEST" ]; then
          if [ -d /models ] && [ -w /models ]; then DEST=/models/${base}-gptq-variant
          elif [ -w "$(dirname "$MODEL_PATH")" ]; then DEST=$(dirname "$MODEL_PATH")/gptq-variant
          else DEST=/tmp/${base}-gptq-variant; fi
        fi
        if [ ! -f "$DEST/quantize_config.json" ]; then
          echo "entrypoint: deriving gptq-config variant -> $DEST"
          python3 /opt/b70/make_gptq_variant.py "$MODEL_PATH" "$DEST"
          # The container runs as root; hand the derived dir to the owner of
          # its parent so the host user can manage/delete it.
          chown -R --reference="$(dirname "$DEST")" "$DEST" 2>/dev/null || true
        fi
        MODEL_PATH=$DEST
      fi
    fi
    # Everything needed is local now; avoid Hub round-trips at serve time.
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
    ;;
  fp8)
    MODEL_PATH=$MODEL
    ;;
esac

ARGS=(
  "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --tensor-parallel-size "${TP:-2}"
  --attention-backend FLASH_ATTN
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${MAX_NUM_SEQS:-8}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --language-model-only
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8000}"
)
[ "${PREFIX_CACHING:-1}" = "0" ] && ARGS+=(--no-enable-prefix-caching)
[ "$PRESET" = "fp8" ] && ARGS+=(--quantization fp8)
[ "$PRESET" = "int4-mtp" ] && ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
[ "${ENABLE_THINKING:-1}" = "0" ] && ARGS+=(--default-chat-template-kwargs '{"enable_thinking":false}')

# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS:-})
echo "entrypoint: preset=$PRESET model=$MODEL_PATH FI_PROVIDER=$FI_PROVIDER HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset}"
exec vllm serve "${ARGS[@]}" "${EXTRA[@]}" "$@"
