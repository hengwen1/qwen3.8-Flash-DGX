#!/usr/bin/env python3
"""Marked greedy generation for backend A/B comparison.
Usage: python3 gen_marked.py <outfile> — writes '<PROMPT>i>...</PROMPTi>' blocks."""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:18300/v1/chat/completions"
PROMPTS = [
    "Explain what a semaphore is in two sentences.",
    "Write a Python function that reverses a linked list.",
    "What is the capital of Australia and why is it not Sydney?",
    "Summarize the plot of Moby-Dick in three sentences.",
    "Convert 47 degrees Celsius to Fahrenheit and show the steps.",
    "Name three uses of quantization in LLM inference.",
]

out = open(sys.argv[1], "w")
for i, p in enumerate(PROMPTS):
    body = json.dumps({
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": p}],
        "max_tokens": 220, "temperature": 0, "reasoning_effort": "low",
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    text = (m.get("reasoning") or "") + "\x01" + (m.get("content") or "")
    out.write(f"<PROMPT{i}>{text}</PROMPT{i}>\n")
    print(f"prompt {i} done ({len(text)} chars)", file=sys.stderr)
out.close()
