#!/usr/bin/env python3
"""Opt-in tcgen05 kernel paths on Jetson Thor (sm_110a, compute capability 11.0).

vLLM v0.31 steers every Blackwell-class tcgen05 kernel with
``is_device_capability_family(100)`` (sm_10x only) or ``major == 10``, so on a
Thor (major 11) the engine silently picks slower generic fallbacks: the
FlashAttention-first priority list instead of FlashInfer's trtllm-gen
attention, a non-trtllm MoE for the NVFP4 routed experts, the Triton GDN
prefill kernel, and cuBLAS for the MoE router GEMM. The wheel itself already
ships sm_110f cubins (CMake builds ``11.0f`` on CUDA >= 13), so the kernels
are there — only the gates keep them off.

This patch extends four of those gates to capability family 110 behind runtime
environment checks. Inert by default and on any other GPU: the injected
expressions require ``VLLM_THOR_TCGEN05=1`` AND family 110. serve.sh sets the
variable automatically on detected Thor boxes; disable per part with
``VLLM_THOR_TCGEN05_<PART>=0``:

  ATTN    platforms/cuda.py::_get_backend_priorities
          FlashInfer (trtllm-gen tcgen05 attention) first for causal non-MLA.
          If FlashInfer misbehaves: VLLM_THOR_TCGEN05_ATTN=0 (or the stock
          VLLM_ATTENTION_BACKEND=FLASH_ATTN).
  MOE     fused_moe/experts/trtllm_nvfp4_moe.py::_supports_current_device
          trtllm-gen NVFP4 fused MoE for the routed experts. The biggest win:
          the experts dominate this model's FLOPs.
  GDN     mamba/gdn/qwen_gdn_linear_attn.py GDN backend selection
          FlashInfer GDN prefill on sm_110a. No CuteDSL here — the in-tree
          CuteDSL GDN kernel is SM100-only (upstream's own comment).
  ROUTER  fused_moe/router/gate_linear.py::is_blackwell
          bf16x3 cuTe-DSL router GEMM (JIT-compiled, tcgen05 MMA). First thing
          to turn off (VLLM_THOR_TCGEN05_ROUTER=0) if a Thor boot fails.

Usage:
  python3 patch_thor_tcgen05.py <cuda.py> <trtllm_nvfp4_moe.py> \
      <qwen_gdn_linear_attn.py> <gate_linear.py>          install
  ... off                                                 remove
"""
import ast
import sys

MARKER = "qwen38-flash-dgx thor-tcgen05"

HELPER = '''

# ---- {marker}: opt-in tcgen05 kernel paths on Jetson Thor (sm_110a) ----
def _thor_tcgen05(part: str) -> bool:
    """True when VLLM_THOR_TCGEN05=1 arms the gate and the part is not
    disabled with VLLM_THOR_TCGEN05_<PART>=0. Only meaningful when the
    calling expression also matches capability family 110 (Thor); returns
    False everywhere else, so the patch is inert on GB10/Spark and upstream
    hardware. Defined here because vLLM gates these paths at import time of
    each module, before any engine config exists."""
    import os as _os
    if _os.environ.get("VLLM_THOR_TCGEN05", "") != "1":
        return False
    return _os.environ.get("VLLM_THOR_TCGEN05_" + part, "1") != "0"
# ---- end {marker} ----
'''

# (file key, helper-anchor, patch-anchor, replacement)
PATCHES = [
    # 1. Attention backend priorities: treat capability 11.0 like SM100 for
    # causal, non-MLA attention (this model is not MLA).
    (
        "cuda.py",
        "logger = init_logger(__name__)",
        "        if device_capability.major == 10 and not use_non_causal:",
        "        if (\n"
        "            device_capability.major == 10\n"
        "            or (_thor_tcgen05(\"ATTN\") and device_capability.major == 11)\n"
        "        ) and not use_non_causal:",
    ),
    # 2. trtllm-gen NVFP4 fused MoE (tcgen05) for the routed experts.
    (
        "trtllm_nvfp4_moe.py",
        "from vllm.platforms import current_platform",
        "        return (\n"
        "            p.is_cuda()\n"
        "            and p.is_device_capability_family(100)\n"
        "            and has_flashinfer_trtllm_fused_moe()\n"
        "        )",
        "        return (\n"
        "            p.is_cuda()\n"
        "            and (\n"
        "                p.is_device_capability_family(100)\n"
        "                or (\n"
        "                    _thor_tcgen05(\"MOE\")\n"
        "                    and p.is_device_capability_family(110)\n"
        "                )\n"
        "            )\n"
        "            and has_flashinfer_trtllm_fused_moe()\n"
        "        )",
    ),
    # 3. FlashInfer GDN prefill backend on sm_110a (linear-attention layers).
    (
        "qwen_gdn_linear_attn.py",
        "from vllm.platforms import current_platform",
        "        # The in-tree CuteDSL kernel targets SM100 only, so it stays off here.\n"
        "        supports_flashinfer = True\n"
        "\n"
        "    if backend in [\"flashinfer\", \"auto\"] and supports_flashinfer:",
        "        # The in-tree CuteDSL kernel targets SM100 only, so it stays off here.\n"
        "        supports_flashinfer = True\n"
        "    elif (\n"
        "        _thor_tcgen05(\"GDN\")\n"
        "        and current_platform.is_device_capability_family(110)\n"
        "        and head_k_dim == 128\n"
        "        and current_platform.get_cuda_runtime_major() >= 13\n"
        "    ):\n"
        "        # thor-tcgen05: FlashInfer GDN prefill on sm_110a (Jetson Thor).\n"
        "        # No CuteDSL either — that kernel is SM100-only.\n"
        "        supports_flashinfer = True\n"
        "\n"
        "    if backend in [\"flashinfer\", \"auto\"] and supports_flashinfer:",
    ),
    # 4. MoE router GEMM tiers (the bf16x3 cuTe-DSL one uses tcgen05 MMA).
    (
        "gate_linear.py",
        "from vllm.platforms import current_platform",
        "        is_blackwell = current_platform.is_device_capability_family(100)",
        "        is_blackwell = current_platform.is_device_capability_family(100) or (\n"
        "            _thor_tcgen05(\"ROUTER\")\n"
        "            and current_platform.is_device_capability_family(110)\n"
        "        )",
    ),
]

HELPER = HELPER.replace("{marker}", MARKER)
HELPER_LINE_1 = "# ---- " + MARKER + ": opt-in tcgen05 kernel paths on Jetson Thor (sm_110a) ----"
HELPER_END = "# ---- end " + MARKER + " ----"


def install(source: str, helper_anchor: str, patch_anchor: str, replacement: str,
            path: str = "") -> str:
    if HELPER_LINE_1 in source:
        # Idempotent: installed replacements contain their anchor text (the
        # GDN one wraps it), so test the replacement first.
        if replacement in source:
            return source
        if patch_anchor in source:
            raise SystemExit("thor-tcgen05: helper present but gate drifted: " + patch_anchor[:60])
        raise SystemExit("thor-tcgen05: helper present but gate expression not found in " + path)
    if source.count(helper_anchor) != 1:
        raise SystemExit("thor-tcgen05: helper anchor not unique: " + helper_anchor)
    if source.count(patch_anchor) != 1:
        raise SystemExit("thor-tcgen05: gate anchor not unique: " + patch_anchor[:60])
    source = source.replace(helper_anchor, helper_anchor + HELPER)
    # The GDN anchor is contained in its replacement; patch it before the
    # helper marker could ever appear inside an anchor (it does not).
    return source.replace(patch_anchor, replacement)


def remove(source: str, helper_anchor: str, replacement: str, patch_anchor: str) -> str:
    # Exact inverse of install: the helper was inserted as anchor + HELPER.
    out = source.replace(helper_anchor + HELPER, helper_anchor)
    if replacement in out:
        out = out.replace(replacement, patch_anchor)
    return out


def main() -> None:
    args = [a for a in sys.argv[1:]]
    off = args and args[-1] == "off"
    paths = args[:-1] if off else args
    if len(paths) != len(PATCHES):
        raise SystemExit(__doc__)
    for (key, helper_anchor, patch_anchor, replacement), path in zip(PATCHES, paths):
        source = open(path).read()
        if off:
            new = remove(source, helper_anchor, replacement, patch_anchor)
            state = "REMOVED" if new != source else "not installed"
        else:
            new = install(source, helper_anchor, patch_anchor, replacement, path)
            state = "INSTALLED" if new != source else "already installed"
        if new != source:
            open(path, "w").write(new)
        ast.parse(new)  # syntax gate; aborts the docker build on drift
        print(f"  thor-tcgen05 {key}: {state} (inert unless VLLM_THOR_TCGEN05=1)")


if __name__ == "__main__":
    main()
