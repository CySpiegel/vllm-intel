# cyspiegel/vllm-xpu-b70 — vLLM tuned for dual Intel Arc B-series (BMG / XPU).
# Unofficial build from github.com/CySpiegel/vllm-intel + CySpiegel/vllm-xpu-kernels.
#
# Two-step build from the vllm-intel repo root:
#   1. base (vLLM fork built for XPU by the repo's own CI Dockerfile):
#        docker build -f docker/Dockerfile.xpu --target vllm-openai -t cyspiegel/vllm-xpu-b70:base .
#   2. this overlay (custom kernel wheel + presets + entrypoint). docker/b70/wheels/
#      is git-ignored: `mkdir -p docker/b70/wheels` and drop the wheel there first.
#        docker build -f docker/b70/Dockerfile \
#          --build-arg KERNEL_WHEEL=<wheel filename placed in docker/b70/wheels/> \
#          --build-arg VLLM_REV=$(git rev-parse --short HEAD) \
#          --build-arg KERNELS_REV=<kernels sha> \
#          -t cyspiegel/vllm-xpu-b70:latest .
ARG BASE_IMAGE=cyspiegel/vllm-xpu-b70:base
FROM ${BASE_IMAGE}

ARG KERNEL_WHEEL
ARG VLLM_REV=unknown
ARG KERNELS_REV=unknown
RUN test -n "${KERNEL_WHEEL}" || { echo "KERNEL_WHEEL build-arg is required" >&2; exit 1; }

# Custom kernels: GDN mixed-batch spec-decode fixes, #544 conv-state reconcile,
# XE2 delta-epilogue OOB fix, spec conv-state register write-back.
COPY docker/b70/wheels/${KERNEL_WHEEL} /tmp/${KERNEL_WHEEL}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps --force-reinstall /tmp/${KERNEL_WHEEL} && rm -f /tmp/${KERNEL_WHEEL}

# Validated serving environment (bare-metal recipe, see docker/b70/README.md).
# oneCCL's libfabric providers ship inside the torch/oneccl wheels under /opt/venv/lib.
# CCL_ZE_IPC_EXCHANGE=pidfd: used when the container has CAP_SYS_PTRACE; under docker's
# default seccomp profile oneCCL falls back to drmfd, which needs /dev/dri/by-path
# bind-mounted (see docker-compose.yaml). No --privileged required either way.
# Do NOT set CCL_ENABLE_SYCL_KERNELS=0 (stock intel/vllm compose): it selects an
# all-reduce that cannot be recorded into XPU graphs and aborts graph capture.
# VLLM_USE_V2_MODEL_RUNNER=0: the XPU stack is validated on the V1 model runner;
# architectures that default to V2 (e.g. Gemma 4) hit an XPU-graph event error in
# V2's structured-output warmup. Override with -e VLLM_USE_V2_MODEL_RUNNER=1.
ENV FI_PROVIDER_PATH=/opt/venv/lib \
    TRITON_INTEL_DEVICE_ARCH=bmg \
    VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    CCL_ZE_IPC_EXCHANGE=pidfd

COPY docker/b70/entrypoint.sh /entrypoint.sh
COPY docker/b70/make_gptq_variant.py /opt/b70/make_gptq_variant.py
RUN chmod +x /entrypoint.sh

LABEL org.opencontainers.image.title="vllm-xpu-b70" \
      org.opencontainers.image.source="https://github.com/CySpiegel/vllm-intel" \
      org.opencontainers.image.description="vLLM for 2x Intel Arc B-series (BMG): INT4/FP8 Qwen3.8-27B presets, MTP spec-decode, custom GDN kernels. Unofficial." \
      org.opencontainers.image.revision="${VLLM_REV}" \
      dev.cyspiegel.vllm-xpu-kernels.revision="${KERNELS_REV}"

# Weights are not baked. Mount checkpoints at /models and/or the HF cache at
# /root/.cache/huggingface; MODEL may be a local path or an HF repo id.
VOLUME ["/models", "/root/.cache/huggingface"]
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
