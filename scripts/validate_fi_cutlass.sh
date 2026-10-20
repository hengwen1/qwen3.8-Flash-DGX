#!/usr/bin/env bash
# Post-reboot validation: FLASHINFER_CUTLASS NvFp4 MoE vs stock VLLM_CUTLASS.
# Usage: bash validate_fi_cutlass.sh          (run after the host has rebooted)
set -u
cd "$(dirname "$0")/.."
PORT=18300
CACHE="$HOME/.cache/qwen38-flash-dgx-compile"

wait_health() {  # up to ~20 min
  for _ in $(seq 1 80); do
    code=$(curl -s -m 3 -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null)
    [ "$code" = 200 ] && return 0
    sleep 15
  done
  return 1
}

gen() {  # gen <outfile> — 6 fixed prompts, greedy
  local out="$1" p
  : > "$out"
  for p in "Explain what a semaphore is in two sentences." \
           "Write a Python function that reverses a linked list." \
           "What is the capital of Australia and why is it not Sydney?" \
           "Summarize the plot of Moby-Dick in three sentences." \
           "Convert 47 degrees Celsius to Fahrenheit and show the steps." \
           "Name three uses of quantization in LLM inference."; do
    curl -s -m 300 "http://localhost:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"qwen3.8-flash-next\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":220,\"temperature\":0,\"reasoning_effort\":\"low\"}" \
      | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
    c = d["choices"][0]["message"]
    print((c.get("reasoning") or "") + "|" + (c.get("content") or ""))
except Exception as e:
    print("ERR", e)' >> "$out"
  done
}

echo "== phase 1: FLASHINFER_CUTLASS (VLLM_THOR_TCGEN05=1, MOE=1) =="
COMPILE_CACHE="$CACHE" TCGEN05=1 VLLM_THOR_TCGEN05_MOE=1 ./scripts/serve.sh || exit 1
wait_health || { echo "engine never became healthy"; docker logs qwen38-flash 2>&1 | tail -40; exit 1; }
docker logs qwen38-flash 2>&1 | grep "NvFp4 MoE backend" | tail -1
gen /tmp/out_fi.txt

echo "== phase 2: stock VLLM_CUTLASS (MOE off) =="
COMPILE_CACHE="$CACHE" TCGEN05=1 VLLM_THOR_TCGEN05_MOE=0 ./scripts/serve.sh || exit 1
wait_health || { echo "engine never became healthy"; exit 1; }
docker logs qwen38-flash 2>&1 | grep "NvFp4 MoE backend" | tail -1
gen /tmp/out_stock.txt

echo "== diff (first divergence per prompt) =="
paste -d'\n' /dev/null /dev/null >/dev/null
same=0; diffn=0
python3 - <<'EOF'
fi = open("/tmp/out_fi.txt").read().splitlines()
st = open("/tmp/out_stock.txt").read().splitlines()
for i, (a, b) in enumerate(zip(fi, st)):
    if a == b:
        print(f"prompt {i}: IDENTICAL")
    else:
        # show first differing position
        j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        print(f"prompt {i}: DIFFERS at char {j}")
        print(f"   fi   : ...{a[max(0,j-40):j+60]!r}")
        print(f"   stock: ...{b[max(0,j-40):j+60]!r}")
EOF

echo "== phase 3: quick decode perf on FLASHINFER_CUTLASS =="
COMPILE_CACHE="$CACHE" TCGEN05=1 VLLM_THOR_TCGEN05_MOE=1 ./scripts/serve.sh || exit 1
wait_health || exit 1
timeout 600 python3 bench/perf.py --port "$PORT" 2>/dev/null || \
  echo "(bench/perf.py needs its usual args — run it manually; see README)"
