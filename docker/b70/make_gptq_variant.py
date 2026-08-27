# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Derive a plain-GPTQ-config serve variant from an auto_round-format export.

The AutoRound export (packing_format auto_round:auto_gptq) already stores
GPTQ-packed tensors; only the config differs. vLLM routes quant_method
"auto-round" through the INC/ARK kernels (slow at batch on XPU) and
quant_method "gptq" through the XPUwNa16 int4 GEMM path. This links the
shards into a new directory and rewrites the config; tensors are untouched.

Usage: make_gptq_variant.py <src_dir> <dest_dir>
"""

import json
import os
import struct
import sys
from pathlib import Path

UNQUANT_DTYPES = {"F16", "BF16", "F32"}
EXCLUDED_PATTERNS = ("in_proj_a", "in_proj_b", "visual", "lm_head", "mtp.")


def safetensors_keys_dtypes(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return {k: v["dtype"] for k, v in header.items() if k != "__metadata__"}


def block_suffix(module: str) -> str | None:
    """'model.language_model.layers.5.self_attn.q_proj' -> 'self_attn.q_proj'."""
    parts = module.split(".layers.")
    if len(parts) < 2:
        return None
    rest = parts[-1].split(".", 1)
    return rest[1] if len(rest) == 2 and rest[0].isdigit() else None


def uniform_block_suffixes(
    quant_modules: set[str], unquant_modules: set[str]
) -> list[str]:
    """Per-block suffixes that are quantized in every layer where they exist.

    Returns [] when some suffix is quantized in one layer and kept in float in
    another (vLLM's per-module scan is then the safer description).
    """
    quant = {s for m in quant_modules if (s := block_suffix(m))}
    unquant = {s for m in unquant_modules if (s := block_suffix(m))}
    if quant & unquant:
        return []
    return sorted(quant)


def link_or_symlink(src: Path, dest: Path) -> None:
    """Hardlink when on the same filesystem, else absolute symlink (the HF
    cache and /models are usually different mounts inside a container)."""
    real = src.resolve()
    try:
        os.link(real, dest)
    except OSError:
        os.symlink(real, dest)


def main(src: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)

    src_cfg = json.loads((src / "config.json").read_text())
    qc = src_cfg["quantization_config"]
    assert qc["quant_method"] == "auto-round", qc["quant_method"]
    assert qc["packing_format"] == "auto_round:auto_gptq", qc["packing_format"]
    assert qc["sym"] is True and qc["bits"] == 4, (qc["bits"], qc["sym"])

    gptq_qc = {
        "quant_method": "gptq",
        "bits": qc["bits"],
        "group_size": qc["group_size"],
        "sym": qc["sym"],
        "desc_act": False,
        "lm_head": False,
        "provider": "auto-round",
    }
    src_cfg["quantization_config"] = gptq_qc

    linked = 0
    for f in sorted(src.iterdir()):
        if f.name in ("config.json", "quantization_config.json", "quant_manifest.json"):
            continue
        if f.name.startswith("."):
            continue
        target = dest / f.name
        if target.exists() or target.is_symlink():
            target.unlink()
        link_or_symlink(f, target)
        linked += 1

    quant_modules, unquant_modules = set(), set()
    for shard in dest.glob("*.safetensors"):
        for key, dtype in safetensors_keys_dtypes(shard).items():
            module, leaf = key.rsplit(".", 1)
            if leaf == "qweight":
                quant_modules.add(module)
            elif leaf == "weight" and dtype in UNQUANT_DTYPES:
                unquant_modules.add(module)

    # vLLM's GPTQ path otherwise derives modules_in_block_to_quantize by
    # scanning the shards, which lists modules by full name. Architectures
    # whose layers legitimately lack a fused shard (Gemma 4 full-attention
    # layers have no v_proj; V is loaded from K) then fail the "all shards of
    # a fused layer quantized" check. Emitting per-block suffixes makes the
    # check uniform across layers.
    suffixes = uniform_block_suffixes(quant_modules, unquant_modules)
    if suffixes:
        gptq_qc["modules_in_block_to_quantize"] = suffixes
    else:
        print(
            "NOTE: mixed-precision suffixes found; leaving module list to vLLM's scan"
        )

    bad = [m for m in quant_modules for p in EXCLUDED_PATTERNS if p in m]
    print(f"linked {linked} files -> {dest}")
    print(
        f"quantized modules: {len(quant_modules)}, unquantized: {len(unquant_modules)}"
    )
    if bad:
        print(f"FAIL: excluded-pattern modules found quantized: {bad[:5]}")
        return 1
    if not any("mlp.gate_proj" in m for m in quant_modules):
        print("FAIL: expected quantized MLP modules missing")
        return 1
    # Config files are written last: quantize_config.json doubles as the
    # "derivation complete and validated" sentinel for restarts.
    (dest / "config.json").write_text(json.dumps(src_cfg, indent=2))
    (dest / "quantize_config.json").write_text(json.dumps(gptq_qc, indent=2))
    print("OK: dtype-scan derivation matches the exclusion recipe")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
