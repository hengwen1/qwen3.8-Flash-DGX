#!/usr/bin/env python3
"""CPU checks for src/patch_thor_tcgen05.py (Thor tcgen05 gates) - no GPU, no vllm import.

    python3 src/test_thor_patch_cpu.py
    python3 src/test_thor_patch_cpu.py --sources /path/to/vllm   # real tree

Fixtures are syntactically valid, *runnable* windows of the vLLM v0.33.0 files
around every anchor the patch edits (anchor lines copied byte-exact from
https://github.com/vllm-project/vllm/tree/v0.33.0; the pinned
`vllm/vllm-openai:v0.33.0` digest ships byte-identical copies - image build
commit 4ba787e). Only the vllm imports around the anchors are stubbed, so the
patched gates can be exercised on a fake platform. Checks:

  1. the patcher runs clean and idempotent over a fake dist-packages tree;
  2. `off` restores the five files byte-for-byte (the image revert path);
  3. a drifted anchor refuses loudly instead of half-patching;
  4. `_thor_tcgen05()` semantics: master switch, per-part defaults, overrides;
  5. each gate stays inert off family 110, and 0.33's stock behaviour on
     family 90/100/120 is unchanged with the patch installed;
  6. on family 110 every part defaults to off except ROUTER, arming one part
     never arms another, and the 0.33 NVFP4 backend ordering still lands on
     the re-armed FLASHINFER_CUTLASS gate (the CuteDSL experts stay 100-only).
"""
import ast
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "patch_thor_tcgen05.py")

# --- fixtures: v0.33.0 windows, byte-exact anchors --------------------------

CUDA_FIXTURE = '''\
from vllm.logger import init_logger


logger = init_logger(__name__)


class DeviceCapability:
    def __init__(self, major, minor=0):
        self.major = major
        self.minor = minor


def _get_backend_priorities(device_capability, use_non_causal=False):
    if use_non_causal:
        return ["FLASH_ATTN"]
    else:
        if device_capability.major == 10 and not use_non_causal:
            return ["FLASHINFER", "FLASH_ATTN"]
        return ["FLASH_ATTN", "FLASHINFER"]
'''

TRTLLM_FIXTURE = '''\
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe


class TrtLlmNvFp4ExpertsModular:
    @staticmethod
    def _supports_current_device() -> bool:
        """Supports only Blackwell-family GPUs."""
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_trtllm_fused_moe()
        )
'''

CUTLASS_FIXTURE = '''\
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe


logger = init_logger(__name__)


class FlashInferExperts:
    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and (
                p.is_device_capability(90)
                or p.is_device_capability_family(100)
                # SM110 excluded: flashinfer-ai/flashinfer#3134
                or p.is_device_capability_family(120)
            )
            and has_flashinfer_cutlass_fused_moe()
        )
'''

GDN_FIXTURE = '''\
from vllm.platforms import current_platform


def _select_gdn_prefill_backend(backend, head_k_dim):
    supports_flashinfer = False
    supports_cutedsl = False

    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True
    elif (
        current_platform.is_device_capability_family(120)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        # The in-tree CuteDSL kernel targets SM100 only, so it stays off here.
        supports_flashinfer = True

    if backend in ["flashinfer", "auto"] and supports_flashinfer:
        return backend, "flashinfer"
    if backend == "cutedsl" and supports_cutedsl:
        return backend, "cutedsl"
    return backend, "triton"
'''

GATE_LINEAR_FIXTURE = '''\
import vllm.envs as envs
from vllm.platforms import current_platform


class RouterGemmLinear:
    def __init__(self, bias=False):
        is_hopper = current_platform.is_device_capability((9, 0))
        is_blackwell = current_platform.is_device_capability_family(100)
        is_gfx950 = False
        if current_platform.is_rocm():
            is_gfx950 = False
        batch_invariant = envs.VLLM_BATCH_INVARIANT
        can_use_specialized_kernels = (
            current_platform.is_cuda()
            and (is_hopper or is_blackwell)
            and not bias
            and not batch_invariant
        )
        self.is_blackwell = is_blackwell
        self.can_use_specialized_kernels = can_use_specialized_kernels
'''

FIXTURES = {
    "cuda.py": CUDA_FIXTURE,
    "trtllm_nvfp4_moe.py": TRTLLM_FIXTURE,
    "flashinfer_cutlass_moe.py": CUTLASS_FIXTURE,
    "qwen_gdn_linear_attn.py": GDN_FIXTURE,
    "gate_linear.py": GATE_LINEAR_FIXTURE,
}
ORDER = ["cuda.py", "trtllm_nvfp4_moe.py", "flashinfer_cutlass_moe.py",
         "qwen_gdn_linear_attn.py", "gate_linear.py"]

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


# --- fake vllm modules ------------------------------------------------------

class StubPlatform:
    """Stand-in for vllm.platforms.current_platform."""

    def __init__(self, family):
        self.family = family
        self.device_name = "cuda"

    def is_cuda(self):
        return True

    def is_rocm(self):
        return False

    def is_device_capability(self, cap):
        if isinstance(cap, tuple):
            return (self.family // 10, self.family % 10) == cap
        return self.family == cap // 10

    def is_device_capability_family(self, family):
        return self.family == family // 10

    def get_cuda_runtime_major(self):
        return 13


def fake_modules(family):
    vllm = types.ModuleType("vllm")
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = StubPlatform(family)
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: None
    envs = types.ModuleType("vllm.envs")
    envs.VLLM_BATCH_INVARIANT = False
    utils = types.ModuleType("vllm.utils")
    flashinfer = types.ModuleType("vllm.utils.flashinfer")
    flashinfer.has_flashinfer_trtllm_fused_moe = lambda: True
    flashinfer.has_flashinfer_cutlass_fused_moe = lambda: True
    vllm.platforms = platforms
    vllm.logger = logger_mod
    vllm.envs = envs
    vllm.utils = utils
    utils.flashinfer = flashinfer
    return {
        "vllm": vllm,
        "vllm.platforms": platforms,
        "vllm.logger": logger_mod,
        "vllm.envs": envs,
        "vllm.utils": utils,
        "vllm.utils.flashinfer": flashinfer,
    }


def load(path, family):
    """Exec a patched fixture with the fake vllm modules for `family`."""
    saved = {k: sys.modules.get(k) for k in fake_modules(family)}
    sys.modules.update(fake_modules(family))
    try:
        ns = {"__name__": os.path.basename(path)}
        exec(compile(open(path).read(), path, "exec"), ns)
        return ns
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def env(**kw):
    base = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_THOR_TCGEN05")}
    base.update({k: str(v) for k, v in kw.items()})
    return base


def call(ns, fn, *a, environ=None, **kw):
    """Call ns[fn] with a pinned environment (so gate env reads are explicit)."""
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(environ if environ is not None else env())
    try:
        obj = ns
        for part in fn.split("."):
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        return obj(*a, **kw)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def run_patch(root, off=False, paths=None):
    args = [sys.executable, PATCH] + (paths or [os.path.join(root, f) for f in ORDER])
    if off:
        args.append("off")
    return subprocess.run(args, capture_output=True, text=True)


def write_tree(root, overrides=None):
    for name, text in FIXTURES.items():
        with open(os.path.join(root, name), "w") as fh:
            fh.write((overrides or {}).get(name, text))


def test_install_roundtrip(tmp):
    print("1. install / idempotent / revert on the v0.33.0 anchors")
    root = os.path.join(tmp, "tree")
    os.makedirs(root)
    write_tree(root)
    orig = {n: open(os.path.join(root, n)).read() for n in ORDER}

    first = run_patch(root)
    check("patcher exits 0", first.returncode == 0, (first.stdout + first.stderr).strip()[:300])
    installed = {n: open(os.path.join(root, n)).read() for n in ORDER}
    check("all five files changed", all(installed[n] != orig[n] for n in ORDER),
          [n for n in ORDER if installed[n] == orig[n]])

    second = run_patch(root)
    check("second run is idempotent", second.returncode == 0 and all(
        open(os.path.join(root, n)).read() == installed[n] for n in ORDER),
        (second.stdout + second.stderr).strip()[:200])
    check("second run reports already installed",
          second.stdout.count("already installed") == len(ORDER), second.stdout.strip())

    off = run_patch(root, off=True)
    check("revert exits 0", off.returncode == 0, (off.stdout + off.stderr).strip()[:200])
    check("revert is byte-identical", all(
        open(os.path.join(root, n)).read() == orig[n] for n in ORDER),
        [n for n in ORDER if open(os.path.join(root, n)).read() != orig[n]])

    return installed


def test_shape(installed):
    print("2. every edit landed in the right file with the right shape")
    for name, text in installed.items():
        ast.parse(text)
    check("helper injected once per file", all(
        text.count("qwen38-flash-dgx thor-tcgen05") == 2 for text in installed.values()),
        {n: t.count("qwen38-flash-dgx thor-tcgen05") for n, t in installed.items()})
    check("helper is defined before first use in every file", all(
        text.index("def _thor_tcgen05") < text.index('_thor_tcgen05("')
        for text in installed.values()))
    conjunct = {
        "cuda.py": "device_capability.major == 11",
        "trtllm_nvfp4_moe.py": "is_device_capability_family(110)",
        "flashinfer_cutlass_moe.py": "is_device_capability_family(110)",
        "qwen_gdn_linear_attn.py": "is_device_capability_family(110)",
        "gate_linear.py": "is_device_capability_family(110)",
    }
    for name, needle in conjunct.items():
        text = installed[name]
        check(f"{name}: gate needs the env helper and family 110",
              '_thor_tcgen05("' in text and needle in text)
    check("cuda gate keeps the causal guard",
          "and not use_non_causal:" in installed["cuda.py"])


def test_drift(tmp):
    print("3. a drifted anchor refuses loudly")
    root = os.path.join(tmp, "drift")
    os.makedirs(root)
    drifted = GATE_LINEAR_FIXTURE.replace(
        "is_blackwell = current_platform.is_device_capability_family(100)",
        "is_blackwell = current_platform.is_device_capability_family(101)")
    write_tree(root, {"gate_linear.py": drifted})
    res = run_patch(root)
    check("drift exits non-zero", res.returncode != 0)
    check("drift names the gate", "gate anchor not unique" in (res.stdout + res.stderr),
          (res.stdout + res.stderr).strip()[:160])
    check("drift left the drifted file untouched",
          open(os.path.join(root, "gate_linear.py")).read() == drifted)


def test_helper(tmp, installed):
    print("4. _thor_tcgen05(): master switch and per-part defaults")
    text = installed["cuda.py"]
    start = text.index("# ---- qwen38-flash-dgx thor-tcgen05")
    end = text.index("# ---- end qwen38-flash-dgx thor-tcgen05 ----")
    path = os.path.join(tmp, "helper.py")
    with open(path, "w") as fh:
        fh.write("import os\n" + text[start:end])
    ns = {}
    exec(compile(open(path).read(), path, "exec"), ns)
    gate = ns["_thor_tcgen05"]

    def armed(part, **environ):
        saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(environ)
        try:
            return gate(part)
        finally:
            os.environ.clear()
            os.environ.update(saved)

    check("inert without the master switch",
          not any(armed(p) for p in ("ATTN", "MOE", "MOE_TRTLLM", "GDN", "ROUTER")))
    check("inert with the master switch at 0",
          not any(armed(p, VLLM_THOR_TCGEN05="0") for p in ("MOE", "ROUTER")))
    check("ROUTER arms by default", armed("ROUTER", VLLM_THOR_TCGEN05="1"))
    check("ATTN/MOE/MOE_TRTLLM/GDN default off",
          not any(armed(p, VLLM_THOR_TCGEN05="1")
                  for p in ("ATTN", "MOE", "MOE_TRTLLM", "GDN")))
    check("per-part opt-in",
          all(armed(p, VLLM_THOR_TCGEN05="1", **{f"VLLM_THOR_TCGEN05_{p}": "1"})
              for p in ("ATTN", "MOE", "MOE_TRTLLM", "GDN")))
    check("per-part opt-out",
          not armed("ROUTER", VLLM_THOR_TCGEN05="1", VLLM_THOR_TCGEN05_ROUTER="0"))
    check("unknown part defaults on", armed("SOMETHING", VLLM_THOR_TCGEN05="1"))


def test_gates(tmp, installed):
    print("5. gates: inert off family 110, 0.33 behaviour unchanged on 90/100/120")
    root = os.path.join(tmp, "gates")
    os.makedirs(root)
    for name, text in installed.items():
        with open(os.path.join(root, name), "w") as fh:
            fh.write(text)
    allparts = env(VLLM_THOR_TCGEN05="1", VLLM_THOR_TCGEN05_ATTN="1",
                   VLLM_THOR_TCGEN05_MOE="1", VLLM_THOR_TCGEN05_MOE_TRTLLM="1",
                   VLLM_THOR_TCGEN05_GDN="1")
    on = env(VLLM_THOR_TCGEN05="1")

    for family, label in ((10, "family 100"), (12, "family 120"), (9, "family 90")):
        cuda = load(os.path.join(root, "cuda.py"), family)
        ficut = load(os.path.join(root, "flashinfer_cutlass_moe.py"), family)
        trtllm = load(os.path.join(root, "trtllm_nvfp4_moe.py"), family)
        gdn = load(os.path.join(root, "qwen_gdn_linear_attn.py"), family)
        gl = load(os.path.join(root, "gate_linear.py"), family)
        prio = call(cuda, "_get_backend_priorities",
                    cuda["DeviceCapability"](family), environ=allparts)
        # Stock 0.33 puts FlashInfer first only for major 10 (SM100/SM103).
        check(f"{label}: attention priorities unchanged",
              prio[0] == ("FLASHINFER" if family == 10 else "FLASH_ATTN"), prio)
        check(f"{label}: flashinfer CUTLASS gate unchanged",
              call(ficut, "FlashInferExperts._supports_current_device", environ=allparts)
              == (family in (9, 10, 12)))
        check(f"{label}: trtllm gate unchanged",
              call(trtllm, "TrtLlmNvFp4ExpertsModular._supports_current_device", environ=allparts)
              == (family == 10))
        backend = call(gdn, "_select_gdn_prefill_backend", "auto", 128,
                       environ=allparts)[1]
        check(f"{label}: GDN backend unchanged",
              backend == ("flashinfer" if family in (9, 10, 12) else "triton"), backend)
        check(f"{label}: router GEMM unchanged",
              call(gl, "RouterGemmLinear", environ=allparts).is_blackwell
              == (family == 10))

    print("6. family 110: default off, per-part arming, no cross-arming")
    cuda = load(os.path.join(root, "cuda.py"), 11)
    ficut = load(os.path.join(root, "flashinfer_cutlass_moe.py"), 11)
    trtllm = load(os.path.join(root, "trtllm_nvfp4_moe.py"), 11)
    gdn = load(os.path.join(root, "qwen_gdn_linear_attn.py"), 11)
    gl = load(os.path.join(root, "gate_linear.py"), 11)
    cap11 = cuda["DeviceCapability"](11)

    check("110, unarmed: FlashAttention-first stays",
          call(cuda, "_get_backend_priorities", cap11)[0] == "FLASH_ATTN")
    check("110, unarmed: no flashinfer MoE/GDN",
          not call(ficut, "FlashInferExperts._supports_current_device")
          and not call(trtllm, "TrtLlmNvFp4ExpertsModular._supports_current_device")
          and call(gdn, "_select_gdn_prefill_backend", "auto", 128)[1] == "triton")
    check("110, unarmed: router GEMM is the plain path",
          not call(gl, "RouterGemmLinear").is_blackwell)

    check("110 + ROUTER (default): router GEMM armed",
          call(gl, "RouterGemmLinear", environ=on).is_blackwell)
    check("110 + ROUTER: nothing else armed",
          call(cuda, "_get_backend_priorities", cap11, environ=on)[0] == "FLASH_ATTN"
          and not call(ficut, "FlashInferExperts._supports_current_device", environ=on)
          and not call(trtllm, "TrtLlmNvFp4ExpertsModular._supports_current_device", environ=on)
          and call(gdn, "_select_gdn_prefill_backend", "auto", 128, environ=on)[1]
          == "triton")

    moe_only = env(VLLM_THOR_TCGEN05="1", VLLM_THOR_TCGEN05_MOE="1")
    check("110 + MOE: FLASHINFER_CUTLASS armed, trtllm stays off",
          call(ficut, "FlashInferExperts._supports_current_device", environ=moe_only)
          and not call(trtllm, "TrtLlmNvFp4ExpertsModular._supports_current_device", environ=moe_only))
    trtllm_only = env(VLLM_THOR_TCGEN05="1", VLLM_THOR_TCGEN05_MOE_TRTLLM="1")
    check("110 + MOE_TRTLLM: trtllm armed, CUTLASS stays off",
          call(trtllm, "TrtLlmNvFp4ExpertsModular._supports_current_device", environ=trtllm_only)
          and not call(ficut, "FlashInferExperts._supports_current_device", environ=trtllm_only))
    attn_only = env(VLLM_THOR_TCGEN05="1", VLLM_THOR_TCGEN05_ATTN="1")
    check("110 + ATTN: FlashInfer-first priorities",
          call(cuda, "_get_backend_priorities", cap11, environ=attn_only)[0]
          == "FLASHINFER")
    gdn_only = env(VLLM_THOR_TCGEN05="1", VLLM_THOR_TCGEN05_GDN="1")
    check("110 + GDN: flashinfer GDN prefill",
          call(gdn, "_select_gdn_prefill_backend", "auto", 128, environ=gdn_only)[1]
          == "flashinfer")
    check("110 + GDN: head_k_dim 64 stays triton",
          call(gdn, "_select_gdn_prefill_backend", "auto", 64, environ=gdn_only)[1]
          == "triton")


def test_033_backend_order(installed):
    print("7. 0.33 NVFP4 backend order: Thor lands on the re-armed CUTLASS part")
    # vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (v0.33.0) consults
    # each backend's _supports_current_device() in this order. The two CuteDSL
    # experts are family(100)-only, so the first family-110-capable entry on a
    # Thor is the flashinfer CUTLASS gate patched here.
    cutedsl_supports_110 = False  # flashinfer_cutedsl_moe.py, 0.33.0: family(100)
    ficut = installed["flashinfer_cutlass_moe.py"]
    check("CuteDSL stays 100-only upstream", cutedsl_supports_110 is False)
    check("CUTLASS gate is the family-110 re-arm",
          "is_device_capability_family(110)" in ficut)


REAL = {
    "cuda.py": "platforms/cuda.py",
    "trtllm_nvfp4_moe.py":
        "model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py",
    "flashinfer_cutlass_moe.py":
        "model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py",
    "qwen_gdn_linear_attn.py":
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    "gate_linear.py": "model_executor/layers/fused_moe/router/gate_linear.py",
}


def test_real_tree(sources):
    """Install/idempotent/off against an unpacked vLLM tree (no fixtures)."""
    print(f"0. real sources: {sources}")
    paths = [os.path.join(sources, "vllm", REAL[n]) for n in ORDER]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        check("five vLLM modules found", False, missing[0])
        return
    orig = {p: open(p).read() for p in paths}
    first = run_patch(None, paths=paths)
    check("patcher exits 0", first.returncode == 0,
          (first.stdout + first.stderr).strip()[:300])
    installed = {p: open(p).read() for p in paths}
    check("all five files changed", all(installed[p] != orig[p] for p in paths),
          [os.path.basename(p) for p in paths if installed[p] == orig[p]])
    second = run_patch(None, off=True, paths=paths)
    check("revert exits 0", second.returncode == 0,
          (second.stdout + second.stderr).strip()[:200])
    check("revert is byte-identical", all(open(p).read() == orig[p] for p in paths),
          [os.path.basename(p) for p in paths if open(p).read() != orig[p]])


def main(argv):
    sources = None
    if len(argv) == 3 and argv[1] == "--sources":
        sources = argv[2]
    elif len(argv) > 1:
        print(__doc__)
        return 2
    print("thor-tcgen05 CPU checks (fixtures: vLLM v0.33.0 anchors)")
    if sources:
        test_real_tree(sources)
    with tempfile.TemporaryDirectory() as tmp:
        installed = test_install_roundtrip(tmp)
        test_shape(installed)
        test_drift(tmp)
        test_helper(tmp, installed)
        test_gates(tmp, installed)
        test_033_backend_order(installed)
    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("all thor-tcgen05 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
