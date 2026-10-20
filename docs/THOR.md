# Jetson Thor (sm_110a): the tcgen05 recipe

The same recipe runs on a **Jetson Thor** (compute capability 11.0, `sm_110a`) with two
arch-specific switches. Both are automatic on detected Thor hardware and inert on a GB10,
so one image and one checkout serve both boxes:

- **The image is built for the GPU it runs on.** `./flash setup` reads the compute
  capability and builds with `--build-arg GPU_ARCH=110a` (Spark boxes keep `121a`; export
  `GPU_ARCH=` to override). That compiles the deterministic top-k kernel as `sm_110a`, and
  raises the FLA Triton shared-memory gate from the GB10's 99 KiB to Thor's full 227 KiB
  Blackwell budget — the big GDN tiles fit whole instead of being split.
- **The tcgen05 kernel paths arm themselves at serve time.** Every Blackwell-class tcgen05
  kernel is gated on sm_10x-only checks (`is_device_capability_family(100)`, `major == 10`),
  so a Thor silently falls back to slower generic kernels even though the wheel already
  ships `sm_110f` cubins. Dockerfile patch 24 re-wires four of those gates behind an env
  check (`VLLM_THOR_TCGEN05=1` AND capability family 110); `scripts/serve.sh` sets that env
  when it detects a Thor (`TCGEN05=auto`, echoed in the startup line; the `thor` profile
  pins it). Per part:

  | env part | upstream gate | what a Thor gains |
  |---|---|---|
  | `MOE` | `trtllm_nvfp4_moe.py::_supports_current_device` | the trtllm-gen **NVFP4 fused MoE** (tcgen05 MMA + TMEM) for the routed experts — this model's dominant FLOPs, until now on the generic MoE path |
  | `ATTN` | `platforms/cuda.py::_get_backend_priorities` | **FlashInfer (trtllm-gen) attention first** for causal non-MLA; the stock major-11 priority list puts FlashAttention first |
  | `GDN` | `qwen_gdn_linear_attn.py` backend pick | the **FlashInfer GDN prefill** kernel instead of Triton/FLA (deliberately no CuteDSL — that kernel is SM100-only upstream) |
  | `ROUTER` | `gate_linear.py::is_blackwell` | the **bf16x3 cuTe-DSL router GEMM** (JIT-compiled tcgen05 MMA) instead of plain cuBLAS |

  Disable any part with `VLLM_THOR_TCGEN05_<ATTN\|MOE\|GDN\|ROUTER>=0` — `serve.sh` passes
  those straight into the container, and the master switch is `TCGEN05=0`. On a Spark the
  patched gates see capability 12.1 and stay off. Two Thor notes: the 128 GB pool and the
  ~75 GiB weight math are the same as the Spark's; the CPU is not, so `CPUSET` defaults to
  all cores there instead of the GB10's Cortex-X5 pin. And Jetson's NVIDIA container runtime
  rejects `--gpus all` (the hook path): `serve.sh` uses `--runtime=nvidia` on detected
  Thor boxes instead — set `DOCKER_GPU_ARGS=` if your install differs. The CPU all of this runs on
  has a knob of its own, outside the container: set the power model as in
  [Power model 4](#power-model-4-inference-the-host-side-knob), which is worth ~5% of the
  concurrent throughput here.

The tcgen05 switches were built and syntax-validated against the v0.31.0 sources and
**validated on real Thor hardware**: the image builds with `GPU_ARCH=110a`; the bf16x3
router GEMM matches an fp32 reference; the trtllm-gen NVFP4 MoE and the FlashInfer GDN
prefill are gated off because their kernels are not available for sm_110 in flashinfer 0.7.0
(forcing them crashes the boot), and therefore default off behind the per-part env vars.
Still: measure with `bench/perf.py` before trusting the numbers, and if a part regresses,
switch it off by env and open an issue.

## Power model 4 (INFERENCE): the host-side knob

Everything above happens inside the container; the other half of a Thor's throughput is set
outside it. A Jetson AGX Thor's `/etc/nvpmodel.conf` defines five power models, and the file's
own `DEFAULT` is **1 (`120W`)**; the usual reflex on a dev kit is `nvpmodel -m 0` (`MAXN`) plus
`jetson_clocks`. For this recipe run **power model 4, `INFERENCE`** — a custom entry added to this
device's conf, tuned from the 120W model, whose comment says the whole thing: *"GPU at the
OC3-clean ceiling (1395MHz), CPU/video/PVA capped to keep total SoC power under the thermal
envelope (avoids MAXN throttle oscillation)"*.

```bash
sudo nvpmodel -m 4        # applies live between models 0/1/4 — all three keep 14 cores online
nvpmodel -q               # -> "NV Power Mode: INFERENCE" and "4"
nvpmodel -q --verbose | grep -E 'PARAM (GPU|EMC|CPU_AE_0):'   # REAL_VAL must equal CONF_VAL
```

| model | CPU cores | CPU ceiling | GPU ceiling | EMC | what it is for |
|---|---|---|---|---|---|
| `0` MAXN | 14 | 2601 MHz | uncapped (sits at 1575 MHz) | 4266 MHz | nothing constrained: fastest clocks, highest power draw |
| `1` 120W | 14 | 2601 MHz | 1386 MHz | 4266 MHz | the conf's `DEFAULT` — workable, just not the tuned setting |
| **`4` INFERENCE** | **14** | **2268 MHz** | **1395 MHz** | **4266 MHz** | full core count and full memory clock, GPU pinned at the ceiling that runs clean, video/PVA down (this recipe uses neither) |
| `2`/`3` 90W/70W | **12** | 2601 / 1998 MHz | 1530 MHz | 4266 MHz | take two cores offline (`nvpmodel` asks for a reboot to do it) — wrong for a recipe that gathers the PLE table on the CPU |

Measured on the dev kit this recipe was validated on — 128 GB, driver 595.78, the server exactly
as `./flash serve thor` starts it (hybrid, 500k, MTP=2, `SEQS=8`, `GPU_MEM=0.80`, chunked prefill
at 8192, `VLLM_THOR_TCGEN05=1`), box otherwise idle, 2026-10-19. Three rounds of 8 streams × 900
tokens (thinking off, `bench/common.py` prompts), cold prefills of a 216 KB filler block (43.6k
prompt tokens, a random nonce in front so the prefix cache cannot hit), and clocks and power
sampled at 0.5 s from devfreq and `hwmon`:

| setting | single-stream decode | 8-stream per-stream, 3 rounds | cold prefill 43.6k | SoC power med/max | Tj max |
|---|---|---|---|---|---|
| model 4 as configured | 46.0 tok/s | 19.8 · 20.1 · 19.4 | 2,206 tok/s | 80 / 98 W | 60 °C |
| model 4 + CPU ceiling raised to 2601 MHz | — | 20.9 · 21.9 · 20.8 | — | 79 / 94 W | 59 °C |
| model 4 + CPU 2601 + `jetson_clocks` | — | 21.2 · 20.6 · 20.7 | — | 81 / 98 W | 59 °C |
| model 1 (`120W`) | 45.9 | 20.5 · 21.2 · 20.3 | 2,199 | 80 / 91 W | 61 °C |
| model 0 (`MAXN`) | 47.1 | 20.5 · 22.3 · 22.0 | 2,258 | 84 / 101 W | 64 °C |

Three things come out of this. **The GPU ceiling is nearly free**: MAXN's 180 MHz of extra
headroom is worth ~2% single-stream (47.1 vs 46.0 tok/s) and ~2% prefill (2,258 vs 2,206), and at
equal CPU ceiling the 8-stream rounds overlap within their spread (rows 2 and 5) — a step here is
memory- and CPU-bound, not FLOP-bound. **The CPU ceiling is the one real concession**: 2268 MHz is invisible
single-stream but costs ~5% of the concurrent throughput, because every step reads 16 rows of the
mmapped table on the CPU ([the PLE gather](HOW-IT-WORKS.md#the-lever)) — so lift it right after
selecting the model:

```bash
sudo nvpmodel -m 4
sudo sh -c 'for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do echo 2601000 > $c; done'
```

**and `jetson_clocks` adds nothing** on top of that (rows 2 and 3 are the same within noise): it
pins min=max *inside* whatever ceilings the model set, so it does not undo `nvpmodel -m 4` — it
just removes the down-DVFS that keeps an idle box quiet. Leave it off unless you measured a reason.
One honest caveat about the reason mode 4 exists: the *"MAXN throttle oscillation"* its conf
comment warns about did **not** reproduce in these runs — through the sustained rounds the GPU
never fell below its cap in any model (0/285, 0/265 and 0/262 samples below cap) and Tj stayed
≤ 64 °C, so treat the ceiling as insurance for a hotter room or a closed
chassis, not as measured performance. Any later `nvpmodel -m …` rewrites the ceilings, so keep the
two lines in that order. To survive reboots, the unit on this box (without the `jetson_clocks`
line its first version had, and without one local `Before=`):

```ini
# /etc/systemd/system/jetson-perf.service
[Unit]
Description=Jetson power model 4 (INFERENCE)
Requires=nvpmodel.service
After=nvpmodel.service

[Service]
Type=oneshot
RemainAfterExit=yes
# nvpmodel fails if the GPU driver has not finished probing (no gpu-gpc-0 devfreq node yet)
ExecStartPre=/bin/sh -c 'i=0; while [ $i -lt 30 ] && [ ! -e /sys/class/devfreq/gpu-gpc-0/available_frequencies ]; do sleep 1; i=$((i+1)); done'
ExecStart=/bin/sh -c 'for i in 1 2 3 4; do /usr/sbin/nvpmodel -m 4 && exit 0; sleep 5; done; /usr/sbin/nvpmodel -m 4'
ExecStart=/bin/sh -c 'for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do echo 2601000 > $c; done'

[Install]
WantedBy=multi-user.target
```

`sudo systemctl daemon-reload && sudo systemctl enable --now jetson-perf`, then check with
`nvpmodel -q --verbose` — on a box where the retry raced the driver the model can silently stay
wherever it was. These are host settings, not container ones: `serve.sh` and `flash` never touch
them, and one image checkout on two machines still needs this done once per machine.
