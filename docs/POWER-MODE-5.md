# Power model 5 (`INFERENCE_10C_MAXN_GPU`): uncapped GPU, ten cores

Companion to [README → Power model 4 (INFERENCE): the host-side knob](../README.md#power-model-4-inference-the-host-side-knob).
Everything here is a **host** setting: `./flash` and `scripts/serve.sh` never touch it, and a
second Thor running the same checkout needs it done again once per machine.

Mode 4 and mode 5 are two custom entries in this device's `/etc/nvpmodel.conf`, both tuned from
the conf's own `DEFAULT` (`1` = `120W`). They answer the same question — how to keep a Thor off
NVIDIA's overcurrent/frequency-cap flags while running *this* recipe — with opposite trades:

| model | name | cores online | CPU ceiling | GPU ceiling | EMC | why |
|---|---|---|---|---|---|---|
| `0` | `MAXN` | 14 | 2601 MHz | uncapped → sits at **1575 MHz** | 4266 MHz | nothing constrained; trips OC2/OC3 under load |
| `1` | `120W` | 14 | 2601 MHz | 1386 MHz | 4266 MHz | the conf's `DEFAULT` |
| `4` | `INFERENCE` | 14 | 2268 MHz | **1395 MHz** (OC3-clean) | 4266 MHz | keep all 14 cores, cap the GPU |
| **`5`** | **`INFERENCE_10C_MAXN_GPU`** | **10** (`CORE_0-9`) | 2268 MHz | **uncapped → 1575 MHz** | 4266 MHz | keep the GPU at its top, drop 4 cores |

## Why mode 5 exists

An uncapped GPU (`GPU MAX_FREQ -1`) sits at its 1575 MHz top under load. With **14 cores online**
that draw trips soctherm's OC2/OC3 overcurrent events — measured on this box on 2026-10-20 at
roughly **1,500 `oc3` events/minute** during sustained load, i.e. the part is being current-capped
repeatedly rather than running free. Mode 4's 1395 MHz GPU ceiling is the point where that count
goes to **0 events/min** and is the setting the README documents and benchmarks.

Mode 5 is the other way to the same goal: leave the GPU uncapped and remove the *simultaneously
switching* CPU cores instead. Four cores offline (`CORE_10-13`) lowers the di/dt peak the power
rails see, which is what the OC flags trip on, and the GPU keeps its full 180 MHz of headroom over
mode 4. The conf says this in its own comment:

```
# Custom: mode 4 (INFERENCE) with MAXN-style uncapped GPU and only CORE_0-9 online.
# CPU_AE clock params for the offlined cores are omitted (like modes 2/3) so a
# live switch does not fail reading their cpufreq nodes.
# NOTE: at MAXN the GPU sits at its 1575 MHz top, which trips soctherm OC2/OC3
# overcurrent events under load (~1500 oc3/min at 14 cores, 2026-10-20);
# mode 4's 1395 MHz cap is the OC3-clean point (0 events/min).
```

## The conf entry

Annotated condensation of what was added to this box's `/etc/nvpmodel.conf`; the verbatim entry is
at the end of this section.

```
< POWER_MODEL ID=5 NAME=INFERENCE_10C_MAXN_GPU >
CPU_ONLINE CORE_0 1  ... CPU_ONLINE CORE_9 1      # 10 cores up
CPU_ONLINE CORE_10 0 ... CPU_ONLINE CORE_13 0     # 4 cores down
CPU_AE_0..9 MIN_FREQ 972000 MAX_FREQ 2268000       # same 2268 MHz ceiling as mode 4
CPU_AE_10..13: <absent>                            # deliberately omitted, see below
GPU MIN_FREQ 315000000 / MAX_FREQ -1               # -1 = no ceiling -> 1575 MHz under load
VIDEO MIN_FREQ 315000000 / MAX_FREQ 945000000
GPU_POWER_GATING GPU_PG_MASK 64
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
PVA0_VPS 810000000 / PVA0_AXI 612000000
EMC MAX_FREQ 4266000000                            # full memory clock, as in 0/1/4
```

Two details worth understanding before editing that conf:

- **`CPU_AE_10..13` are left out on purpose** (the same convention as the stock 90W/70W models).
  `nvpmodel` applies a model by writing every param it lists; writing a cpufreq node belonging to
  an offline core fails, so an entry that still carried those params would make the switch error
  out part-way through. Only the online cores get clock params.
- **`GPU MAX_FREQ -1` is "no ceiling"**, not "no frequency". The devfreq then climbs to whatever
  `available_frequencies` tops out at — 1575 MHz here.

<details>
<summary>The entry as added (verbatim)</summary>

```
# Custom: mode 4 (INFERENCE) with MAXN-style uncapped GPU and only CORE_0-9 online.
# CPU_AE clock params for the offlined cores are omitted (like modes 2/3) so a
# live switch does not fail reading their cpufreq nodes.
# NOTE: at MAXN the GPU sits at its 1575 MHz top, which trips soctherm OC2/OC3
# overcurrent events under load (~1500 oc3/min at 14 cores, 2026-10-20);
# mode 4's 1395 MHz cap is the OC3-clean point (0 events/min).
< POWER_MODEL ID=5 NAME=INFERENCE_10C_MAXN_GPU >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 1
CPU_ONLINE CORE_3 1
CPU_ONLINE CORE_4 1
CPU_ONLINE CORE_5 1
CPU_ONLINE CORE_6 1
CPU_ONLINE CORE_7 1
CPU_ONLINE CORE_8 1
CPU_ONLINE CORE_9 1
CPU_ONLINE CORE_10 0
CPU_ONLINE CORE_11 0
CPU_ONLINE CORE_12 0
CPU_ONLINE CORE_13 0
GPU_POWER_GATING GPU_PG_MASK 64
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
CPU_AE_0 MIN_FREQ 972000
CPU_AE_0 MAX_FREQ 2268000
CPU_AE_1 MIN_FREQ 972000
CPU_AE_1 MAX_FREQ 2268000
CPU_AE_2 MIN_FREQ 972000
CPU_AE_2 MAX_FREQ 2268000
CPU_AE_3 MIN_FREQ 972000
CPU_AE_3 MAX_FREQ 2268000
CPU_AE_4 MIN_FREQ 972000
CPU_AE_4 MAX_FREQ 2268000
CPU_AE_5 MIN_FREQ 972000
CPU_AE_5 MAX_FREQ 2268000
CPU_AE_6 MIN_FREQ 972000
CPU_AE_6 MAX_FREQ 2268000
CPU_AE_7 MIN_FREQ 972000
CPU_AE_7 MAX_FREQ 2268000
CPU_AE_8 MIN_FREQ 972000
CPU_AE_8 MAX_FREQ 2268000
CPU_AE_9 MIN_FREQ 972000
CPU_AE_9 MAX_FREQ 2268000
GPU MIN_FREQ 315000000
GPU MAX_FREQ -1
VIDEO MIN_FREQ 315000000
VIDEO MAX_FREQ 945000000
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
PVA0_VPS MAX_FREQ 810000000
PVA0_AXI MAX_FREQ 612000000
EMC MAX_FREQ 4266000000
```

</details>

## Applying it

```bash
sudo nvpmodel -m 5
nvpmodel -q                                    # -> "NV Power Mode: INFERENCE_10C_MAXN_GPU" / 5
nvpmodel -q --verbose | grep -E 'PARAM (GPU|EMC|CPU_AE_0):'   # REAL_VAL must equal CONF_VAL
cat /sys/devices/system/cpu/online             # -> 0-9
nproc                                          # -> 10
```

**Expect this one to ask for a reboot**, unlike the 0/1/4 switching that applies live. Models that
change the *online core set* are the ones `nvpmodel` cannot always apply to a running kernel — that
is already true of the stock `2`/`3` entries (see the README's table), and mode 5 moves four cores
instead of two. If `nvpmodel -q` reports mode 5 but `/sys/devices/system/cpu/online` still says
`0-13`, the core offline never landed and the box is running mode 5's clocks with mode 0/4's
topology: **the configuration this document is about to measure does not exist**. Reboot and recheck
both lines before believing any number.

To raise the CPU ceiling the way the README recommends for mode 4 (the 2268 MHz cap costs ~5% of
concurrent throughput, because every step gathers 16 rows of the mmapped PLE table on the CPU),
note that four of the `cpufreq` nodes may be gone:

```bash
sudo sh -c 'for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do echo 2601000 > "$c" 2>/dev/null || true; done'
```

And keep the order: any later `nvpmodel -m …` rewrites the ceilings.

## What the recipe gains and loses — not measured

The GPU-ceiling half of this trade is already quantified on this box, by the mode 0 vs mode 4 rows
in the README: **+180 MHz of GPU is worth ~2% single-stream decode (47.1 vs 46.0 tok/s) and ~2%
cold prefill (2,258 vs 2,206 tok/s)**, at +4 W median SoC power and +4 °C `Tj`.

Treat that ~2% as a floor, and as evidence that is going stale. Those rows were measured on
2026-10-19, one day **before** the FLASHINFER_CUTLASS NVFP4 MoE became the Thor default
(2026-10-20, +18% cold prefill / +4% decode — see the README's tcgen05 table). A faster MoE kernel
moves the same wall-clock budget into the GPU, which is exactly the resource mode 5 uncaps; the
gap between "capped at 1395" and "free at 1575" is plausibly wider now than the 2% measured against
the slower kernel path. Re-measure mode 4 vs mode 5 on the current default serve before reusing the
2%.

The core-count half is **not** quantified, and there is no mode-5 benchmark to quote. The prior
says it hurts: this recipe's decode step is CPU- and memory-bound, not FLOP-bound, and the CPU side
is exactly what loses four cores. Two mechanisms, both easy to check:

- `WORKERS` defaults to 32 gather threads (`VLLM_PLE_MMAP_WORKERS`, [HOW-IT-WORKS](HOW-IT-WORKS.md))
  and was tuned on the GB10's 10 performance cores. On a 10-core Thor the fault-overlap trick that
  buys +8%/+17% there (README, `FAST_ROWS=0`) is oversubscribed 3.2:1 against cores that also carry
  the engine's own CPU work.
- A container pinned to cores that are offline gets its topology silently broken. On Thor `CPUSET`
  defaults to empty (all cores), which follows the online set and is safe. If you set `CPUSET`
  explicitly — the GB10 default `5-9,15-19` is the trap here — pin inside `0-9`:
  `./flash serve thor CPUSET=0-9`.

## Measuring it before using it

Reproduce the README's harness and add the event counter to the sampling loop:

```bash
# the number that decides whether mode 5's premise holds: OC events per minute
sudo journalctl -k -d | grep -ci oc3               # per boot
perf list | grep -i soctherm                       # PMU exposed? then poll during a round:
#   perf stat -e 'soctherm:*' -a sleep 60

# clocks + power at 0.5 s while it runs, as in the README's table
grep . /sys/class/devfreq/gpu-gpc-0/cur_freq       # actually at 1575, or capped somewhere else?
cat /sys/class/hwmon/hwmon*/temp*_input /sys/class/hwmon/hwmon*/power*_input 2>/dev/null

python3 bench/perf.py                              # TTFT, single-stream, concurrency, prefill curve
python3 bench/cold_prefill.py                      # nonce'd prompts: prefill without cache flattery
```

For numbers comparable with the README's table, match its harness: the server as `./flash serve
thor` starts it (hybrid, 500k, MTP=2), with `SEQS=8`, `GPU_MEM=0.80`, chunked prefill at 8192 and
`VLLM_THOR_TCGEN05=1` — those three knobs are the README run's, not the `thor` profile's, so set
them explicitly. Box otherwise idle, thinking off, three rounds of 8 streams × 900 tokens.
`bench/perf.py` has no CLI for that: its rounds come from module defaults
(`concurrency(levels=…, tokens=300)`) and its prefill targets extend one another, which the README
flags as a prefix-cache flattery trap — use `bench/cold_prefill.py` for the prefill line.

Judge the result on three axes, in this order: **OC3 count** (if it is still nonzero, mode 5's
premise is wrong and the GPU is not actually free-running, just capped elsewhere), **per-stream and
aggregate decode** (the four lost cores), **`Tj` max** (the thermal envelope). Then decide: mode 4
if you want the OC3-clean, documented point; mode 5 if the +2% of an uncapped GPU is worth more
than the cores, which needs a number, not a hunch.

To survive reboots, the mode 4 unit in the README works with two changes — the model number, and
dropping the ceiling line if you want to stay at mode 5's own 2268 MHz:

```ini
# /etc/systemd/system/jetson-perf.service
[Unit]
Description=Jetson power model 5 (INFERENCE_10C_MAXN_GPU)
Requires=nvpmodel.service
After=nvpmodel.service

[Service]
Type=oneshot
RemainAfterExit=yes
# nvpmodel fails if the GPU driver has not finished probing (no gpu-gpc-0 devfreq node yet)
ExecStartPre=/bin/sh -c 'i=0; while [ $i -lt 30 ] && [ ! -e /sys/class/devfreq/gpu-gpc-0/available_frequencies ]; do sleep 1; i=$((i+1)); done'
ExecStart=/bin/sh -c 'for i in 1 2 3 4; do /usr/sbin/nvpmodel -m 5 && exit 0; sleep 5; done; /usr/sbin/nvpmodel -m 5'
ExecStart=/bin/sh -c 'for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do echo 2601000 > "$c" 2>/dev/null || true; done'

[Install]
WantedBy=multi-user.target
```

`sudo systemctl daemon-reload && sudo systemctl enable --now jetson-perf`, then verify with
`nvpmodel -q --verbose` **and** `cat /sys/devices/system/cpu/online` — on a box where the retry
raced the driver, or where the core offline needed the reboot, either half can silently stay where
it was.

## Status

Mode 5 is a **configuration hypothesis with one supporting measurement**: the ~1,500 `oc3`/min at
14 uncapped cores (2026-10-20) against 0/min at mode 4's 1395 MHz cap. Its performance on this
recipe is unmeasured, and the mode 4 numbers in the README remain the reference. Keep mode 4 as the
default until the measurements above say otherwise.

Related: [README → Power model 4](../README.md#power-model-4-inference-the-host-side-knob) (the
measured table this trades against), [HOW-IT-WORKS.md](HOW-IT-WORKS.md) (why the CPU is on the
per-token critical path at all).
