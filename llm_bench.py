#!/usr/bin/env python3
"""
llm_bench.py -- context-length performance benchmark for OpenAI-compatible LLM servers.

Measures how an LLM server degrades as the prompt (context) grows:

  * TTFT            time to first token  (prefill + queue latency)
  * prefill tok/s   prompt_tokens / TTFT
  * decode tok/s    completion tokens / generation time  (per stream)
  * ITL             inter-token latency (p50/p95)
  * total latency   whole request
  * aggregate tok/s throughput under concurrency (batch mode)
  * accuracy        "needle in a haystack" retrieval at long context (opt-in)

Pure standard library (urllib) -- no third-party packages required.
numpy/pandas/matplotlib are used only if importable (plots).

Examples
--------
# Sweep context length on a vLLM server, single stream, 3 samples per point:
python3 llm_bench.py --base-url http://192.168.2.38:8000/v1 \
    --model Qwen3.8-Flash-Next --context-lengths 1k,4k,16k,64k,128k \
    --repeats 3 --max-tokens 256 --out runs/qwen

# Throughput under load (4 concurrent requests per point):
python3 llm_bench.py --base-url ... --concurrency 1,2,4 --context-lengths 4k,32k

# Pure prefill/decode timing, no retrieval task, warm prefix cache:
python3 llm_bench.py --base-url ... --no-needle --cache-mode shared --context-lengths 8k,16k

# Reasoning model: disable thinking so max_tokens is spent on the answer:
python3 llm_bench.py --base-url ... --no-think --max-tokens 128

Output: <out>.jsonl (every request), <out>.csv (summary), <out>.md (report),
<out>.json (config+summary), <out>.png (charts, if matplotlib present).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib only)
# --------------------------------------------------------------------------- #


class HttpError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def _urlopen(url: str, payload: Any, headers: Dict[str, str], timeout: float, method: str = "POST"):
    """Open an HTTP response. Returns a file-like object for streaming reads."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    hdrs = {"Accept": "application/json", **headers}
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")[:2000]
        except Exception:  # pragma: no cover
            body = ""
        raise HttpError(f"HTTP {e.code} from {url}: {_short(body)}", e.code, body) from e
    except urllib.error.URLError as e:
        raise HttpError(f"cannot reach {url}: {e.reason}") from e


def http_json(url: str, payload: Any = None, headers: Optional[Dict[str, str]] = None,
              timeout: float = 60.0, method: str = "POST") -> Any:
    resp = _urlopen(url, payload, headers or {}, timeout, method)
    with resp:
        raw = resp.read().decode("utf-8", "replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise HttpError(f"non-JSON response from {url}: {_short(raw)}") from e


def _short(s: str, n: int = 300) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + "..."


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #


def pctl(values: List[float], q: float) -> float:
    """Linear-interpolated percentile; nan for empty input."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return float(vals[0])
    s = sorted(vals)
    k = (len(s) - 1) * max(0.0, min(1.0, q))
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(s[int(k)])
    return float(s[lo] * (hi - k) + s[hi] * (k - lo))


def med(values: List[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(statistics.median(vals)) if vals else float("nan")


def mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(statistics.fmean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

_SAMPLE_FIELDS = dict(
    ctx_target=int, ctx_actual=int, concurrency=int, run=int, ok=bool, error=str,
    ttft_s=float, total_s=float, prefill_tps=float, decode_tps=float,
    itl_p50_ms=float, itl_p95_ms=float, token_events=int,
    prompt_tokens=int, completion_tokens=int, finish=str,
    needle_ok=bool, answer=str, batch_wall_s=float, ts=float,
)


@dataclass
class Sample:
    ctx_target: int = 0
    ctx_actual: int = 0
    concurrency: int = 1
    run: int = 0
    ok: bool = False
    error: str = ""
    ttft_s: float = float("nan")
    total_s: float = float("nan")
    prefill_tps: float = float("nan")
    decode_tps: float = float("nan")
    itl_p50_ms: float = float("nan")
    itl_p95_ms: float = float("nan")
    token_events: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish: str = ""
    needle_ok: Optional[bool] = None
    answer: str = ""
    batch_wall_s: float = float("nan")
    ts: float = 0.0

    @property
    def key(self) -> Tuple[int, int]:
        return (self.ctx_target, self.concurrency)


# --------------------------------------------------------------------------- #
# Backend: talks to one server, knows how to stream one request
# --------------------------------------------------------------------------- #

TEXT_KEYS = ("content", "reasoning", "text", "response")


class Backend:
    """Thin client for OpenAI-compatible (vLLM/TGI/SGLang/llama.cpp) or Ollama servers."""

    def __init__(self, args):
        self.args = args
        self.base_url = args.base_url.rstrip("/")
        self.model = args.model
        self.timeout = args.timeout
        self.headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
        self.kind = args.endpoint  # resolved in probe() when 'auto'
        self._tokenize_url: Optional[str] = None
        self.max_model_len: Optional[int] = None
        self.cpl = 4.0  # chars per token; calibrated from server-reported prompt_tokens

    def note(self, char_len: int, tokens: int) -> None:
        """Feed back a measured (prompt chars, prompt tokens) pair so prompts stay
        correctly sized on servers that expose no tokenizer (e.g. Ollama)."""
        if tokens > 32 and char_len > 200:
            self.cpl = min(12.0, max(1.0, 0.6 * self.cpl + 0.4 * (char_len / tokens)))

    # -- discovery ---------------------------------------------------------- #

    def probe(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"base_url": self.base_url, "model": self.model}
        if self.kind == "auto":
            if self._try(["/models", "/v1/models"], method="GET"):
                self.kind = "chat"
            elif self._try(["/api/tags"], method="GET"):
                self.kind = "ollama"
            else:
                raise HttpError(f"could not find an OpenAI-compatible or Ollama API at {self.base_url}")
        if self.kind in ("chat", "completions"):
            for path in ("/models", "/v1/models"):
                data = self._try([path], method="GET")
                if data:
                    if path == "/v1/models" and not self.base_url.endswith("/v1"):
                        self.base_url = self.base_url.rstrip("/") + "/v1"  # accept base URLs without /v1
                    for m in data.get("data", []):
                        if m.get("id") == self.model or (self.model is None and not self.max_model_len):
                            if self.model is None:
                                self.model = m.get("id")
                            self.max_model_len = m.get("max_model_len") or m.get("max_context_length")
                    break
        elif self.kind == "ollama" and self.model is None:
            data = self._try(["/api/tags"], method="GET")
            if data and data.get("models"):
                self.model = data["models"][0]["name"]
        if self.model is None:
            raise HttpError("no --model given and the server did not advertise one")
        # locate a tokenizer endpoint (optional, used for exact prompt sizing)
        if not self.args.no_tokenize:
            for url in self._tokenize_candidates():
                try:
                    out = http_json(url, {"model": self.model, "prompt": "hello world"},
                                    self.headers, min(self.timeout, 20))
                    if isinstance(out, dict) and ("count" in out or "tokens" in out):
                        self._tokenize_url = url
                        break
                except Exception:
                    continue
        info.update(model=self.model, kind=self.kind, max_model_len=self.max_model_len,
                    tokenizer=bool(self._tokenize_url))
        return info

    def _tokenize_candidates(self) -> List[str]:
        b = self.base_url
        roots = [b, b[:-3] if b.endswith("/v1") else b.rstrip("/")]
        roots += [b + "/v1" if not b.endswith("/v1") else b]
        seen, out = set(), []
        for r in roots:
            for suffix in ("/tokenize", "/v1/tokenize"):
                u = r.rstrip("/") + suffix
                if u not in seen:
                    seen.add(u)
                    out.append(u)
        return out

    def _try(self, paths: List[str], method: str = "POST") -> Optional[Any]:
        for p in paths:
            try:
                return http_json(self.base_url.rstrip("/") + p, None if method == "GET" else {},
                                 self.headers, min(self.timeout, 15), method)
            except Exception:
                continue
        return None

    # -- tokenization ------------------------------------------------------- #

    def count_tokens(self, text: str) -> Optional[int]:
        if self._tokenize_url:
            try:
                out = http_json(self._tokenize_url, {"model": self.model, "prompt": text},
                                self.headers, min(self.timeout, 30))
                if "count" in out:
                    return int(out["count"])
                if "tokens" in out:
                    return len(out["tokens"])
            except Exception:
                self._tokenize_url = None  # stop paying the round trip
        return None  # caller falls back to the char heuristic

    # -- one streamed request ---------------------------------------------- #

    def generate(self, prompt: str, question: str, max_tokens: int, seed: int) -> Dict[str, Any]:
        if self.kind == "chat":
            return self._openai(prompt, question, max_tokens, seed, chat=True)
        if self.kind == "completions":
            return self._openai(prompt, question, max_tokens, seed, chat=False)
        if self.kind == "ollama":
            return self._ollama(prompt + question, max_tokens, seed)
        raise ValueError(f"unknown backend kind {self.kind!r}")

    def _payload_openai(self, prompt: str, question: str, max_tokens: int, seed: int, chat: bool) -> Tuple[str, dict]:
        if chat:
            url = self.base_url + "/chat/completions"
            msgs = []
            if self.args.system:
                msgs.append({"role": "system", "content": self.args.system})
            msgs.append({"role": "user", "content": prompt + question})
            payload = {"model": self.model, "messages": msgs, "max_tokens": max_tokens,
                       "temperature": self.args.temperature, "top_p": self.args.top_p,
                       "seed": seed, "stream": True, "stream_options": {"include_usage": True}}
        else:
            url = self.base_url + "/completions"
            payload = {"model": self.model, "prompt": prompt + question, "max_tokens": max_tokens,
                       "temperature": self.args.temperature, "top_p": self.args.top_p,
                       "seed": seed, "stream": True, "stream_options": {"include_usage": True}}
        if self.args.ignore_eos:
            payload["ignore_eos"] = True
        if self.args.min_tokens:
            payload["min_tokens"] = self.args.min_tokens
        if self.args.no_think:
            payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        if self.args.extra_body:
            payload.update(json.loads(self.args.extra_body))
        return url, payload

    def _openai(self, prompt: str, question: str, max_tokens: int, seed: int, chat: bool) -> Dict[str, Any]:
        url, payload = self._payload_openai(prompt, question, max_tokens, seed, chat)
        if not self.args.stream:
            payload["stream"] = False
            payload.pop("stream_options", None)
            t0 = time.perf_counter()
            out = http_json(url, payload, self.headers, self.timeout)
            total = time.perf_counter() - t0
            ch = (out.get("choices") or [{}])[0]
            text = (ch.get("message") or {}).get("content") if chat else ch.get("text", "")
            usage = out.get("usage") or {}
            return dict(ok=True, ttft=total, total=total, total_no_stream=True,
                        deltas=[], text=text or "",
                        prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                        finish=ch.get("finish_reason") or "")
        return self._read_stream(url, payload, fmt="sse")

    def _ollama(self, prompt: str, max_tokens: int, seed: int) -> Dict[str, Any]:
        url = self.base_url.rstrip("/")
        url = url[:-3] if url.endswith("/v1") else url
        url += "/api/chat"
        payload = {"model": self.model, "stream": self.args.stream,
                   "messages": [{"role": "user", "content": prompt}],
                   "options": {"temperature": self.args.temperature, "top_p": self.args.top_p,
                               "num_predict": max_tokens, "seed": seed}}
        if self.args.extra_body:
            payload.update(json.loads(self.args.extra_body))
        if self.args.no_think:
            payload["think"] = False  # Ollama >= 0.45: disable thinking for thinking models
        if not self.args.stream:
            t0 = time.perf_counter()
            out = http_json(url, payload, self.headers, self.timeout)
            total = time.perf_counter() - t0
            num = lambda k: out[k] if isinstance(out.get(k), int) else None
            text = (out.get("message") or {}).get("content") or out.get("response") or ""
            return dict(ok=True, ttft=float("nan"), total=total, total_no_stream=True, deltas=[],
                        text=text, prompt_tokens=num("prompt_eval_count"), completion_tokens=num("eval_count"),
                        finish=out.get("done_reason") or "",
                        server_prefill_s=(num("prompt_eval_duration") or 0) / 1e9 or None,
                        server_decode_s=(num("eval_duration") or 0) / 1e9 or None, error="")
        return self._read_stream(url, payload, fmt="ndjson")

    def _read_stream(self, url: str, payload: dict, fmt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            resp = _urlopen(url, payload, {**self.headers, "Accept": "text/event-stream" if fmt == "sse"
                                           else "application/x-ndjson"}, self.timeout)
        except HttpError:
            raise
        ttfb = time.perf_counter() - t0
        deltas: List[float] = []
        parts: List[str] = []
        usage: Dict[str, Any] = {}
        finish = ""
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if fmt == "sse":
                    if not line.startswith("data:"):
                        continue
                    line = line[5:].strip()
                    if line == "[DONE]":
                        break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("usage"):
                    for k, v in obj["usage"].items():
                        if isinstance(v, int):
                            usage[k] = v
                for k in ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration"):
                    if isinstance(obj.get(k), int):
                        usage[k] = obj[k]
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = "".join(str(d.get(k) or "") for k in TEXT_KEYS if d.get(k))
                    if not piece:
                        piece = str(ch.get("text") or "")
                    if piece:
                        deltas.append(time.perf_counter() - t0)
                        parts.append(piece)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                # Ollama /api/chat and /api/generate chunk shapes
                if not obj.get("choices"):
                    msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
                    piece = str(msg.get("content") or obj.get("response") or "")
                    if piece:
                        deltas.append(time.perf_counter() - t0)
                        parts.append(piece)
                if obj.get("done") and obj.get("done_reason"):
                    finish = obj["done_reason"]
        total = time.perf_counter() - t0
        ns = lambda k: (usage[k] / 1e9) if isinstance(usage.get(k), int) and usage[k] > 0 else None
        pt = usage.get("prompt_tokens", usage.get("prompt_eval_count"))
        ct = usage.get("completion_tokens", usage.get("eval_count"))
        return dict(ok=True, ttft=deltas[0] if deltas else ttfb, total=total,
                    total_no_stream=False, deltas=deltas, text="".join(parts),
                    prompt_tokens=pt, completion_tokens=ct, finish=finish, error="",
                    server_prefill_s=ns("prompt_eval_duration"), server_decode_s=ns("eval_duration"))


# --------------------------------------------------------------------------- #
# Prompt construction: filler + needle-in-haystack
# --------------------------------------------------------------------------- #

FILLER = [
    "The logistics hub rerouted freight through the northern depot after the bridge closed for repairs.",
    "Analysts noted that quarterly margins improved mostly because of lower energy costs at the smelter.",
    "A calibration run of the spectrometer was logged without anomalies by the night technician.",
    "The committee debated whether the archive should be migrated to cold storage before the audit.",
    "Rainfall measurements from the eastern gauges were cross checked against the satellite estimate.",
    "Maintenance crews replaced the bearing on pump three and recorded a twelve percent drop in vibration.",
    "The manuscript was catalogued under a provisional shelfmark pending review by the head librarian.",
    "Firmware version nine introduced a stricter timeout for the telemetry uplink during congestion.",
    "Pricing for the bulk contract was fixed for two years with an index linked adjustment clause.",
    "Survey teams mapped the shoreline and found the erosion rate had slowed since the previous spring.",
    "The training corpus was deduplicated with a locality sensitive hashing pass before tokenization.",
    "Warehouse inventory showed a discrepancy of three pallets attributable to a mislabelled shipment.",
    "Grid operators shed load briefly while the transformer substation came back online after the fault.",
    "The inspection report mentioned corrosion around the flange but no structural concern was raised.",
    "Crew schedules were rebalanced so that no technician exceeded the regulated fatigue limit.",
    "A second opinion confirmed the dating of the ceramic shards to the late bronze age period.",
    "Sales in the northern region grew after the distributor added the product to its core catalogue.",
    "The benchmark harness repeated each trial fifty times and discarded the slowest two runs.",
    "Water treatment logs recorded turbidity within limits throughout the entire monitoring window.",
    "The vendor published a changelog describing four bug fixes and one reverted optimisation.",
]

NEEDLE_TMPL = ("SECRET ENTRY {n:04d}: the access code for vault {n:04d} is {code}. "
               "This line is unique in the document.")
QUESTION = ("\n\nQuestion: What is the access code for vault {n:04d}? "
            "Answer with only the code, nothing else.")


def _filler_units(rng: random.Random, count: int) -> List[str]:
    """`count` self-contained filler units, each ~100-150 tokens worth of chars,
    with a document header every few units so long prompts look like a real corpus."""
    units: List[str] = []
    doc = rng.randint(1, 500)
    for i in range(count):
        if i % rng.randint(4, 7) == 0:
            doc += rng.randint(1, 7)
            head = f"[DOCUMENT {doc:05d} | SECTION {rng.randint(1, 40)} | PAGE {rng.randint(1, 900)}]\n"
        else:
            head = ""
        sents = " ".join(rng.choice(FILLER) for _ in range(rng.randint(1, 3)))
        units.append(f"{head}{rng.choice('-')} {sents}\n")
    return units


def build_prompt(target_tokens: int, rng: random.Random, needle: bool,
                 depth: float, count_tokens: Optional[Callable[[str], int]] = None,
                 question: str = "", header: str = "", cpl: float = 4.0) -> Tuple[str, str, int]:
    """Return (prompt, code, vault_no).

    Sizes the prompt to ~target_tokens by slicing a stable unit pool and calibrating
    chars-per-token against the server tokenizer (falls back to ~4 chars/token).
    Slicing a stable prefix keeps prompts byte-identical across repeats, so
    --cache-mode shared can actually hit the server's prefix cache.
    """
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
    n = rng.randint(1, 9999)

    pool = _filler_units(rng, max(8, int(target_tokens / 25) + 64))
    avg = max(20.0, sum(len(u) for u in pool) / len(pool))

    def assemble(k: int) -> str:
        k = max(1, min(len(pool), k))
        chosen = pool[:k]
        if needle and k > 1:
            pos = max(1, min(k - 1, int(round(depth * (k - 1)))))
            text = "".join(chosen[:pos]) + NEEDLE_TMPL.format(n=n, code=code) + "\n\n" + "".join(chosen[pos:])
        else:
            text = "".join(chosen)
        return header + text + question

    cpl = max(1.0, cpl)  # chars per token; calibrated by Backend.note() when no tokenizer
    k = max(1, round(target_tokens * cpl / avg))
    best_k, best_err = k, float("inf")
    for _ in range(6):
        text = assemble(k)
        actual = count_tokens(text) if count_tokens else None
        measured = bool(actual)
        actual = actual or max(1, int(len(text) / cpl))
        if abs(actual - target_tokens) < best_err:
            best_k, best_err = k, abs(actual - target_tokens)
        if best_err <= max(4, target_tokens * 0.015):
            break
        if measured:  # trust the tokenizer, then rescale the slice by the observed ratio
            cpl = len(text) / actual
        k_next = max(1, round(k * target_tokens / actual))
        if k_next == k:  # one-unit granularity: nudge toward the target
            k_next = k + (1 if actual < target_tokens else -1)
        if k_next < 1 or k_next == best_k:
            break
        k = k_next
    return assemble(best_k), code, n


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #


def run_point(be: Backend, args, ctx: int, concurrency: int, run: int) -> List[Sample]:
    """Fire `concurrency` identical-context requests at once, repeat index `run`."""
    # 'shared' must be byte-identical across repeats AND runs so the server's prefix
    # cache can be reused; 'bust' varies everything (cold prefill every request).
    shared = args.cache_mode == "shared"
    base_rng = random.Random((args.seed * 1_000_003) ^ (ctx << 8)
                             ^ (0 if shared else (run ^ 977) ^ getattr(args, "invocation_salt", 0)))

    def one(idx: int) -> Sample:
        rng = random.Random(base_rng.randrange(2**31) + (0 if shared else idx * 7919))
        nonce = "" if shared else f"[RUN {run} REQ {idx} NONCE {rng.randrange(2**48):012x}]\n"
        header = (args.system_head + "\n" if args.system_head else "") + nonce
        prompt, code, vault_no = build_prompt(
            ctx, rng, needle=args.needle, depth=args.needle_depth,
            count_tokens=be.count_tokens if not args.no_tokenize else None,
            question="", header=header, cpl=be.cpl)
        q = QUESTION.format(n=vault_no) if args.needle else "\n\nSummarise the document in one sentence."
        s = Sample(ctx_target=ctx, concurrency=concurrency, run=run, ts=time.time())
        try:
            r = be.generate(prompt, q, args.max_tokens, seed=args.seed + run * 131 + idx)
        except HttpError as e:
            s.error, s.ok = str(e), False
            return s
        except Exception as e:  # noqa: BLE001
            s.error, s.ok = f"{type(e).__name__}: {_short(str(e))}", False
            return s
        s.ok = True
        if r.get("error"):
            s.ok, s.error = False, str(r["error"])
            return s
        s.prompt_tokens = int(r.get("prompt_tokens") or 0)
        s.completion_tokens = int(r.get("completion_tokens") or 0)
        be.note(len(prompt), s.prompt_tokens)
        s.ctx_actual = s.prompt_tokens or ctx
        s.finish = r.get("finish") or ""
        s.answer = (r.get("text") or "").strip()[:400]
        s.ttft_s = float(r["ttft"])
        s.total_s = float(r["total"])
        d = [t for t in (r.get("deltas") or [])]
        s.token_events = len(d)
        itls = [(d[i] - d[i - 1]) * 1000 for i in range(1, len(d))]
        s.itl_p50_ms, s.itl_p95_ms = pctl(itls, 0.5), pctl(itls, 0.95)
        n_tok = s.completion_tokens or len(d)
        # prefer the server's own prefill/decode timers when it reports them (Ollama)
        if r.get("server_prefill_s") and s.prompt_tokens:
            s.prefill_tps = s.prompt_tokens / r["server_prefill_s"]
        elif s.ttft_s > 0 and s.prompt_tokens and not r["total_no_stream"]:
            s.prefill_tps = s.prompt_tokens / s.ttft_s
        if r.get("server_decode_s") and n_tok > 1:
            s.decode_tps = n_tok / r["server_decode_s"]
        elif not r["total_no_stream"]:
            gen = s.total_s - s.ttft_s
            if gen > 0 and n_tok > 1:
                s.decode_tps = (n_tok - 1) / gen
        elif n_tok > 1 and s.total_s > 0:  # --no-stream: only end-to-end time is known
            s.decode_tps = n_tok / s.total_s
            s.ttft_s = float("nan")  # not measurable without streaming
        if args.needle and code:
            s.needle_ok = bool(re.search(re.escape(code), s.answer, re.I))
        s.error = ""
        return s

    t0 = time.perf_counter()
    if concurrency == 1:
        samples = [one(0)]
    else:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            samples = list(ex.map(one, range(concurrency)))
    wall = time.perf_counter() - t0
    for s in samples:
        s.batch_wall_s = wall
    return samples


def aggregate(samples: List[Sample]) -> Dict[str, Any]:
    ok = [s for s in samples if s.ok]
    agg: Dict[str, Any] = {
        "ctx_target": samples[0].ctx_target, "concurrency": samples[0].concurrency,
        "n": len(samples), "ok": len(ok),
        "ctx_actual": int(mean([s.ctx_actual for s in ok])) if ok else 0,
        "prompt_tokens": int(mean([s.prompt_tokens for s in ok])) if ok else 0,
        "completion_tokens": int(mean([s.completion_tokens for s in ok])) if ok else 0,
        "ttft_ms_p50": med([s.ttft_s * 1e3 for s in ok]), "ttft_ms_p95": pctl([s.ttft_s * 1e3 for s in ok], 0.95),
        "ttft_ms_max": max([s.ttft_s * 1e3 for s in ok], default=float("nan")),
        "prefill_tps": med([s.prefill_tps for s in ok]),
        "decode_tps_p50": med([s.decode_tps for s in ok]), "decode_tps_p95": pctl([s.decode_tps for s in ok], 0.95),
        "itl_ms_p50": med([s.itl_p50_ms for s in ok]), "itl_ms_p95": med([s.itl_p95_ms for s in ok]),
        "ms_per_tok": (1000.0 / agg_tmp) if (agg_tmp := med([s.decode_tps for s in ok])) and not math.isnan(agg_tmp) else float("nan"),
        "tok_per_chunk": (mean([s.completion_tokens or s.token_events for s in ok]) / mean([s.token_events for s in ok])) if ok and mean([s.token_events for s in ok]) else float("nan"),
        "total_s_p50": med([s.total_s for s in ok]), "total_s_p95": pctl([s.total_s for s in ok], 0.95),
        "batch_wall_s": med([s.batch_wall_s for s in ok]),
        "errors": sorted({s.error for s in samples if s.error})[:3],
    }
    needles = [s.needle_ok for s in ok if s.needle_ok is not None]
    agg["needle_acc"] = (sum(needles) / len(needles)) if needles else None
    if agg["batch_wall_s"] and not math.isnan(agg["batch_wall_s"]) and agg["batch_wall_s"] > 0:
        # one batch = `concurrency` simultaneous streams; batch_wall_s is that batch's wall time
        agg["aggregate_tps"] = agg["completion_tokens"] * agg["concurrency"] / agg["batch_wall_s"]
    else:
        agg["aggregate_tps"] = float("nan")
    return agg


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

F2 = lambda v: "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"
F0 = lambda v: " n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.0f}"
FP = lambda v: " n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100*v:.0f}%"


def fmt_ctx(n: int) -> str:
    return f"{n//1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


def render_table(aggs: List[Dict[str, Any]]) -> str:
    hdr = (f"{'ctx':>7} {'C':>3} {'ok':>5} {'ctxTok':>7} {'ttft p50':>9} {'ttft p95':>9} "
           f"{'prefill tok/s':>14} {'decode tok/s':>13} {'ms/tok':>7} {'chunk p50':>9} {'total p50':>10} "
           f"{'agg tok/s':>10} {'needle':>7}")
    lines = [hdr, "-" * len(hdr)]
    for a in aggs:
        lines.append(
            f"{fmt_ctx(a['ctx_target']):>7} {a['concurrency']:>3} {a['ok']:>2}/{a['n']:<2} "
            f"{a['ctx_actual']:>7} {F0(a['ttft_ms_p50'])+'ms':>9} {F0(a['ttft_ms_p95'])+'ms':>9} "
            f"{F0(a['prefill_tps']):>14} {F2(a['decode_tps_p50']):>13} {F2(a['ms_per_tok']):>7} "
            f"{F2(a['itl_ms_p50']):>9} {F2(a['total_s_p50']):>10} {F0(a['aggregate_tps']):>10} "
            f"{FP(a['needle_acc']):>7}")
    return "\n".join(lines)


def write_outputs(prefix: str, aggs: List[Dict[str, Any]], samples: List[Sample], cfg: dict) -> List[str]:
    written = []
    if not samples:
        return written
    j = prefix + ".jsonl"
    with open(j, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")
        written.append(j)
    c = prefix + ".csv"
    cols = list(aggs[0].keys())
    with open(c, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for a in aggs:
            w.writerow({k: ("" if (isinstance(v, float) and math.isnan(v)) else v) for k, v in a.items()})
        written.append(c)
    md = prefix + ".md"
    with open(md, "w") as f:
        f.write(f"# LLM context-length benchmark\n\n```json\n{json.dumps(cfg, indent=2)}\n```\n\n")
        f.write("```\n" + render_table(aggs) + "\n```\n\n")
        errs = [(a["ctx_target"], a["concurrency"], a["errors"]) for a in aggs if a["errors"]]
        for ctx, cc, e in errs:
            f.write(f"- ctx={fmt_ctx(ctx)} C={cc}: " + " | ".join(e) + "\n")
        written.append(md)
    js = prefix + ".json"
    with open(js, "w") as f:
        json.dump({"config": cfg, "summary": aggs}, f, indent=2, default=str)
        written.append(js)
    png = plot(prefix + ".png", aggs)
    if png:
        written.append(png)
    return written


def plot(png: str, aggs: List[Dict[str, Any]]) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    panels = [("prefill_tps", "prefill tok/s (prompt/TTFT)"), ("decode_tps_p50", "decode tok/s (per stream)"),
              ("ttft_ms_p50", "TTFT ms"), ("total_s_p50", "total latency s"),
              ("aggregate_tps", "aggregate tok/s (all streams)"), ("needle_acc", "needle accuracy")]
    concurrencies = sorted({a["concurrency"] for a in aggs})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ax, (col, title) in zip(axes.flat, panels):
        for cc in concurrencies:
            pts = sorted([a for a in aggs if a["concurrency"] == cc and a["ok"]], key=lambda a: a["ctx_target"])
            xs = [a["ctx_actual"] or a["ctx_target"] for a in pts]
            ys = [a[col] for a in pts]
            keep = [(x, y) for x, y in zip(xs, ys) if y is not None and not (isinstance(y, float) and math.isnan(y))]
            if not keep:
                continue
            ax.plot([k[0] for k in keep], [k[1] for k in keep], marker="o", label=f"C={cc}")
        ax.set_xscale("log", base=2)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("prompt tokens")
        if col == "needle_acc":
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    axes.flat[0].legend(fontsize=8)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("LLM performance vs context length", fontsize=12)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return png


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_sizes(spec: str) -> List[int]:
    out = []
    for tok in re.split(r"[,\s]+", spec.strip()):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmMgG]?)", tok)
        if not m:
            raise SystemExit(f"bad context length {tok!r} (use e.g. 512, 4k, 32768)")
        v = float(m.group(1)) * {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[m.group(2).lower()]
        out.append(int(v))
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark LLM server performance vs context length",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                   help="OpenAI-compatible base URL (…/v1) or Ollama root (…:11434)")
    p.add_argument("--model", default=os.environ.get("LLM_MODEL"), help="model id (default: first served)")
    p.add_argument("--api-key", default=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--endpoint", choices=["auto", "chat", "completions", "ollama"], default="auto",
                   help="chat=/v1/chat/completions, completions=/v1/completions (bypasses chat template)")
    p.add_argument("--context-lengths", default="1k,2k,4k,8k,16k,32k", help="comma list: 512,4k,32k,128k")
    p.add_argument("--concurrency", default="1", help="comma list of concurrent streams per point")
    p.add_argument("--repeats", type=int, default=3, help="sample batches per (context, concurrency) point")
    p.add_argument("--warmup", type=int, default=1, help="unrecorded batches per point")
    p.add_argument("--max-tokens", type=int, default=128, help="tokens to generate per request")
    p.add_argument("--min-tokens", type=int, default=None, help="force equal-length generations")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--timeout", type=float, default=600.0, help="per-request HTTP timeout (s)")
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                   help="stream responses (required for TTFT/ITL)")
    p.add_argument("--ignore-eos", action="store_true",
                   help="always generate max_tokens (vLLM) - makes decode tok/s much more reliable")
    p.add_argument("--no-think", action="store_true",
                   help="suppress reasoning tokens (vLLM chat_template_kwargs / Ollama think=false) "
                        "so --max-tokens is spent on the answer")
    p.add_argument("--extra-body", default=None, help='JSON merged into the request body, e.g. {"top_k":1}')
    p.add_argument("--system", default=None, help="system prompt (chat endpoints)")
    p.add_argument("--system-head", default=None, help="extra text placed before the filler document")
    # prompt / task
    p.add_argument("--needle", dest="needle", action=argparse.BooleanOptionalAction, default=True,
                   help="embed + ask for a secret code (long-context accuracy check)")
    p.add_argument("--needle-depth", type=float, default=0.5, help="0=start, 0.5=middle, 1=end")
    p.add_argument("--cache-mode", choices=["bust", "shared"], default="bust",
                   help="bust: unique nonce per request (cold prefill); shared: identical prompt (warm prefix cache)")
    p.add_argument("--stable-prompts", action="store_true",
                   help="in bust mode, keep prompts identical across invocations (default varies them per "
                        "process so a previous run's KV blocks cannot be reused)")
    p.add_argument("--no-tokenize", action="store_true", help="skip server /tokenize, use 4 chars/token")
    # output
    p.add_argument("--out", default="bench", help="output file prefix (or 'none' to skip files)")
    p.add_argument("--max-context", type=int, default=None, help="cap context sweep (default: server max_model_len - max_tokens)")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first failed point")
    p.add_argument("--estimate-only", action="store_true", help="only report prompt token counts, no generation")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    # Salt prompts per process so 'bust' really means cold prefill: without this a
    # rerun of the same command would hit the server's prefix cache from the last run.
    args.invocation_salt = 0 if (args.stable_prompts or args.cache_mode == "shared") else (time.time_ns() & 0xFFFF_FFFF)
    sizes = parse_sizes(args.context_lengths)
    concurrencies = [int(c) for c in re.split(r"[,\s]+", args.concurrency) if c]

    be = Backend(args)
    if not args.quiet:
        print(f"probing {args.base_url} ...", flush=True)
    info = be.probe()
    if not args.quiet:
        print("  " + json.dumps(info), flush=True)

    cap = args.max_context or (be.max_model_len - args.max_tokens - 8 if be.max_model_len else None)
    if cap:
        dropped = [s for s in sizes if s > cap]
        sizes = [s for s in sizes if s <= cap]
        if dropped:
            print(f"note: skipping {[fmt_ctx(d) for d in dropped]} (over capacity {fmt_ctx(cap)})", flush=True)
    if not sizes:
        print("nothing to benchmark after capacity filtering", file=sys.stderr)
        return 2

    if args.estimate_only:
        print(f"{'ctx':>8} {'tokens':>8} {'chars':>9}")
        print("       (exact if the server exposes /tokenize, else ~chars/4)")
        for s in sizes:
            rng = random.Random(args.seed)
            prompt, _, _ = build_prompt(s, rng, args.needle, args.needle_depth,
                                        be.count_tokens if not args.no_tokenize else None,
                                        "", args.system_head or "", be.cpl)
            n = be.count_tokens(prompt)
            print(f"{fmt_ctx(s):>8} {n if n else int(len(prompt) / 4):>8} {len(prompt):>9}"
                  f"{' exact' if n else ' approx'}")
        return 0

    samples: List[Sample] = []
    t_start = time.perf_counter()
    interrupted = False
    try:
        for cc in concurrencies:
            for ctx in sizes:
                for r in range(-args.warmup, args.repeats):
                    batch = run_point(be, args, ctx, cc, max(r, 0))
                    if r >= 0:
                        samples.extend(batch)
                    if args.fail_fast and not all(s.ok for s in batch):
                        raise KeyboardInterrupt
                agg = aggregate([s for s in samples if s.key == (ctx, cc)])
                line = (f"[C={cc}] ctx={fmt_ctx(ctx):>6} tok={agg['ctx_actual']:>7} "
                        f"ttft={F0(agg['ttft_ms_p50'])}ms prefill={F0(agg['prefill_tps'])}tok/s "
                        f"decode={F2(agg['decode_tps_p50'])}tok/s total={F2(agg['total_s_p50'])}s "
                        f"ok={agg['ok']}/{agg['n']}"
                        + (f" needle={FP(agg['needle_acc'])}" if agg["needle_acc"] is not None else ""))
                if not args.quiet:
                    print(line, flush=True)
                if agg["ok"] == 0 and agg["errors"]:
                    print("       errors: " + " | ".join(agg["errors"]), flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted - reporting completed samples", file=sys.stderr)

    aggs = [aggregate([s for s in samples if s.key == k])
            for k in sorted({s.key for s in samples})]
    print()
    print(render_table(aggs))
    if not args.stream:
        print("note: --no-stream -> TTFT/prefill not measurable; 'decode tok/s' is completion_tokens/total latency")
    if args.cache_mode == "shared":
        print("note: --cache-mode shared sends a byte-identical prompt (warm prefix cache); use --warmup >= 1")
    if args.out and args.out != "none":
        prefix = args.out
        if os.path.isdir(prefix):
            prefix = os.path.join(prefix, time.strftime("bench_%Y%m%d_%H%M%S"))
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        files = write_outputs(prefix, aggs, samples, {
            "base_url": args.base_url, "model": be.model, "endpoint": be.kind,
            "max_model_len": be.max_model_len, "tokenizer": bool(be._tokenize_url),
            "context_lengths": sizes, "concurrency": concurrencies, "repeats": args.repeats,
            "warmup": args.warmup, "max_tokens": args.max_tokens, "temperature": args.temperature,
            "stream": args.stream, "needle": args.needle, "needle_depth": args.needle_depth,
            "cache_mode": args.cache_mode, "stable_prompts": args.stable_prompts, "seed": args.seed, "wall_time_s": round(time.perf_counter() - t_start, 1),
        })
        print("\nwrote: " + ", ".join(files))
    return 1 if interrupted or any(a["ok"] == 0 for a in aggs) else 0


if __name__ == "__main__":
    sys.exit(main())

