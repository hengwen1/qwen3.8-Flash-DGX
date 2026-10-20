#!/usr/bin/env python3
"""Cold-prefill probe, README method: random nonce in front so the prefix
cache cannot hit. One 43.6k-token filler block per measurement.
Usage: python3 cold_prefill.py [n_runs]"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:18300/v1/chat/completions"
FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
          "Meanwhile the engineer refactored the scheduler to reduce tail latency. ")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3

for run in range(n):
    nonce = f"[nonce {run} {time.time_ns()}] "
    body = (FILLER * 1454)[:261600]  # ~43.6k tokens
    prompt = nonce + "Below is a log. Reply with only the word OK.\n\n" + body + "\n\nOK"
    payload = json.dumps({
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    e2e = time.time() - t0
    ptoks = d["usage"]["prompt_tokens"]
    cached = d["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"run {run}: prompt {ptoks} tok (cached {cached}) e2e {e2e:.2f}s "
          f"-> cold prefill ~{ptoks / e2e:.0f} tok/s", flush=True)
