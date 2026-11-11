"""Provider-agnostic LLM client for the hybrid SemVer-bump pipeline.

Each provider implementation takes an instructions string and a state string,
and returns `{"pred": "major|minor|patch", "probs": {"major": p1, ...}}`.
On-disk caching is namespaced by (provider, model) so switching providers
doesn't invalidate the Jev cache and vice-versa.

Supported providers (as of 2026-10-08):
  * typesafe   : the original Jev TypeSafe classifier (langchain-typesafe)
  * openai     : GPT-4o-mini and friends via /v1/chat/completions
  * anthropic  : Claude Haiku/Sonnet via /v1/messages
  * ollama     : any local Ollama model via /api/chat

The OpenAI / Anthropic / Ollama paths all ask for structured JSON output:
  {"label": "<major|minor|patch>", "p_major": 0.x, "p_minor": 0.y, "p_patch": 0.z}
and normalise the probabilities to sum to 1. If the JSON is malformed we
fall back to the string label with sharp (0.9) probabilities on that class.

Usage:
    from llm_clients import make_client
    cli = make_client("openai", model="gpt-4o-mini",
                      cache_path="jev_cache/openai_4omini_v2cargo.json")
    out = cli.predict(instructions, state)
    #   out = {"pred": "minor", "probs": {"major": 0.08, "minor": 0.72, "patch": 0.20}}
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, Optional

CLASSES = ("major", "minor", "patch")


# ---------- base ------------------------------------------------------------

def _normalise_probs(d: Dict[str, float]) -> Dict[str, float]:
    out = {c: max(0.0, float(d.get(c, 0.0))) for c in CLASSES}
    s = sum(out.values())
    if s <= 0:
        return {c: 1.0 / len(CLASSES) for c in CLASSES}
    return {c: v / s for c, v in out.items()}


PARSE_FAILURES = {"json_ok": 0, "regex_fallback": 0, "last_resort": 0}


def _parse_json_response(text: str, fallback_label: Optional[str] = None) -> Dict:
    """Pull `{label, p_major, p_minor, p_patch}` out of model text.
    Tries a strict JSON parse, then a regex fallback.

    If `fallback_label` is given and parsing fails, returns a sharp
    distribution on that label.

    Maintains PARSE_FAILURES stats so a sweep
    can report what fraction of LLM responses were well-formed JSON vs
    needed regex fallback vs last-resort uniform.
    """
    # Pass 1: find first {...} block (greedy)
    m = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    candidates = []
    if m:
        candidates.append(m.group(0))
    candidates.append(text.strip())
    for cand in candidates:
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                label = (d.get("label") or d.get("choice") or d.get("bump")
                         or "").strip().lower()
                probs = {
                    "major": d.get("p_major", d.get("probability_major", 0)),
                    "minor": d.get("p_minor", d.get("probability_minor", 0)),
                    "patch": d.get("p_patch", d.get("probability_patch", 0)),
                }
                if any(float(v) > 0 for v in probs.values()):
                    normalised = _normalise_probs(probs)
                    if label not in CLASSES:
                        label = max(normalised, key=normalised.get)
                    PARSE_FAILURES["json_ok"] += 1
                    return {"pred": label, "probs": normalised}
                if label in CLASSES:
                    probs = {c: (0.9 if c == label else 0.05) for c in CLASSES}
                    PARSE_FAILURES["json_ok"] += 1
                    return {"pred": label, "probs": probs}
        except (ValueError, TypeError):
            continue
    # Pass 2: look for a bare word (regex fallback — real signal degraded)
    low = text.lower()
    for c in CLASSES:
        if re.search(rf"\b{c}\b", low):
            PARSE_FAILURES["regex_fallback"] += 1
            return {"pred": c, "probs": {k: (0.9 if k == c else 0.05) for k in CLASSES}}
    # Last resort (worst case — LLM response unusable)
    label = fallback_label or "patch"
    PARSE_FAILURES["last_resort"] += 1
    return {"pred": label, "probs": {k: (0.9 if k == label else 0.05) for k in CLASSES}}


class LLMClient:
    """Base class — on-disk cache + retry loop + hooks."""

    provider: str = "base"

    def __init__(self, model: str, cache_path: str, max_retries: int = 3,
                 retry_sleep_s: float = 1.0):
        self.model = model
        self.cache_path = cache_path
        self.max_retries = max_retries
        self.retry_sleep_s = retry_sleep_s
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        self._dirty = 0

    # Subclasses implement these ------------------------------------------
    def _call(self, instructions: str, state: str) -> Dict:
        raise NotImplementedError

    # Shared plumbing -----------------------------------------------------
    def _cache_key(self, instructions: str, state: str) -> str:
        h = hashlib.sha1()
        h.update(self.provider.encode()); h.update(b"|")
        h.update(self.model.encode());    h.update(b"|")
        h.update(instructions.encode());  h.update(b"||")
        h.update(state.encode())
        return h.hexdigest()

    def predict(self, instructions: str, state: str) -> Dict:
        key = self._cache_key(instructions, state)
        if key in self.cache:
            return self.cache[key]
        for attempt in range(self.max_retries):
            try:
                out = self._call(instructions, state)
                self.cache[key] = out
                self._dirty += 1
                if self._dirty % 25 == 0:
                    self.flush()
                return out
            except Exception as e:  # noqa: BLE001
                print(f"  [{self.provider}/{self.model}] error ({str(e)[:120]}); "
                      f"retry {attempt+1}/{self.max_retries}", file=sys.stderr)
                time.sleep(self.retry_sleep_s * (attempt + 1))
        return {"pred": "patch", "probs": {c: 1.0 / len(CLASSES) for c in CLASSES}}

    def flush(self):
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.cache, f)
        os.replace(tmp, self.cache_path)


# ---------- shared JSON-asking prompt wrapper ------------------------------

JSON_RESPONSE_SUFFIX = (
    "\n\nRespond with STRICT JSON only (no markdown fences, no commentary) in "
    "exactly this shape:\n"
    '{"label":"major|minor|patch","p_major":0.0,"p_minor":0.0,"p_patch":0.0}\n'
    "The three probabilities must sum to 1.0."
)


def _wrap_prompt(instructions: str, state: str) -> str:
    return (f"{instructions}\n\n"
            f"Release pair brief:\n{state}\n"
            f"{JSON_RESPONSE_SUFFIX}")


# ---------- TypeSafe (Jev) --------------------------------------------------

class TypeSafeClient(LLMClient):
    provider = "typesafe"

    def __init__(self, model: str = "default",
                 cache_path: str = "jev_cache/typesafe.json",
                 criteria: Optional[Dict] = None,
                 default_instructions: str = "", **kw):
        super().__init__(model, cache_path, **kw)
        self.criteria = criteria
        self.default_instructions = default_instructions
        self._load_key()
        from langchain_typesafe import Choice, TypeSafeClassifier
        self.classifier = TypeSafeClassifier()
        self._Choice = Choice

    @staticmethod
    def _load_key():
        if os.environ.get("TYPESAFE_API_KEY"):
            return
        for p in (".typesafe_token", os.path.expanduser("~/.typesafe_token")):
            try:
                t = open(p).read().strip()
                if t:
                    os.environ["TYPESAFE_API_KEY"] = t
                    return
            except FileNotFoundError:
                continue
        sys.exit("ERROR: no TYPESAFE_API_KEY (set env or .typesafe_token).")

    def _call(self, instructions: str, state: str) -> Dict:
        question = {"bump": self._Choice(criteria=self.criteria,
                                         instructions=instructions)}
        resp = self.classifier.invoke({"state": state, "questions": question})
        ans = resp.answers["bump"]
        label = getattr(ans, "choice", None) or getattr(ans, "value", None)
        raw = getattr(ans, "probabilities", None) or {}
        probs = {c: float(raw.get(c, 0.0)) for c in CLASSES}
        probs = _normalise_probs(probs)
        if label not in CLASSES:
            label = max(probs, key=probs.get)
        return {"pred": label, "probs": probs}


# ---------- OpenAI -----------------------------------------------------------

class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str = "gpt-4o-mini",
                 cache_path: Optional[str] = None, **kw):
        cache_path = cache_path or f"jev_cache/openai_{model.replace('/', '-')}.json"
        super().__init__(model, cache_path, **kw)
        self._load_key()
        import httpx
        self.httpx = httpx
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type":  "application/json",
        }

    @staticmethod
    def _load_key():
        if os.environ.get("OPENAI_API_KEY"):
            return
        for p in (".openai_token", os.path.expanduser("~/.openai_token")):
            try:
                t = open(p).read().strip()
                if t:
                    os.environ["OPENAI_API_KEY"] = t
                    return
            except FileNotFoundError:
                continue
        sys.exit("ERROR: no OPENAI_API_KEY (set env or write .openai_token).")

    def _call(self, instructions: str, state: str) -> Dict:
        prompt = _wrap_prompt(instructions, state)
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "max_tokens": 150,
        }
        r = self.httpx.post(self.endpoint, json=body, headers=self.headers,
                            timeout=60.0)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return _parse_json_response(text)


# ---------- Anthropic --------------------------------------------------------

class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str = "claude-haiku-5-5",
                 cache_path: Optional[str] = None, **kw):
        cache_path = cache_path or f"jev_cache/anthropic_{model.replace('/', '-')}.json"
        super().__init__(model, cache_path, **kw)
        self._load_key()
        import httpx
        self.httpx = httpx
        self.endpoint = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key":          os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version":  "2023-06-01",
            "content-type":       "application/json",
        }

    @staticmethod
    def _load_key():
        if os.environ.get("ANTHROPIC_API_KEY"):
            return
        for p in (".anthropic_token", os.path.expanduser("~/.anthropic_token")):
            try:
                t = open(p).read().strip()
                if t:
                    os.environ["ANTHROPIC_API_KEY"] = t
                    return
            except FileNotFoundError:
                continue
        sys.exit("ERROR: no ANTHROPIC_API_KEY (set env or write .anthropic_token).")

    def _call(self, instructions: str, state: str) -> Dict:
        prompt = _wrap_prompt(instructions, state)
        body = {
            "model":       self.model,
            "max_tokens":  200,
            "temperature": 0.0,
            "messages":    [{"role": "user", "content": prompt}],
        }
        r = self.httpx.post(self.endpoint, json=body, headers=self.headers,
                            timeout=60.0)
        r.raise_for_status()
        text = r.json()["content"][0]["text"]
        return _parse_json_response(text)


# ---------- Ollama (local) ---------------------------------------------------

class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, model: str = "llama3.1:8b",
                 cache_path: Optional[str] = None,
                 host: str = "http://127.0.0.1:11434", **kw):
        cache_path = cache_path or f"jev_cache/ollama_{model.replace(':','_').replace('/', '-')}.json"
        super().__init__(model, cache_path, **kw)
        import httpx
        self.httpx = httpx
        self.host = host.rstrip("/")

    def _call(self, instructions: str, state: str) -> Dict:
        prompt = _wrap_prompt(instructions, state)
        body = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "format":  "json",       # Ollama's JSON mode
            "options": {
                "temperature": 0.0,
                "num_predict": 200,
                # Ollama's default num_ctx is 2048, but our few-shot prompts
                # often run 1500-2500 tokens; a too-small context triggers
                # 500 Internal Server Error on generation. Force 4096.
                "num_ctx":     4096,
            },
        }
        r = self.httpx.post(f"{self.host}/api/generate", json=body, timeout=300.0)
        if r.status_code >= 400:
            # Include the server's error body in the exception so the caller's
            # retry/log loop can see WHY Ollama 500-ed (truncation, OOM, etc).
            raise RuntimeError(f"ollama {r.status_code}: {r.text[:300]}")
        text = r.json().get("response", "")
        return _parse_json_response(text)


# ---------- Mock (no API) ----------------------------------------------------

class MockClient(LLMClient):
    """Deterministic rule-based stand-in. For plumbing tests only — not a
    realistic LLM proxy, so never use it for reported numbers."""
    provider = "mock"

    def __init__(self, model: str = "mock",
                 cache_path: str = "jev_cache/mock.json", **kw):
        super().__init__(model, cache_path, **kw)

    def _call(self, instructions: str, state: str) -> Dict:
        s = state.lower()
        if "breaking signals" in s and "none" not in s.split("breaking signals")[1].split("\n")[0]:
            probs = {"major": 0.75, "minor": 0.15, "patch": 0.10}
        elif "addition signals" in s and "none" not in s.split("addition signals")[1].split("\n")[0]:
            probs = {"major": 0.10, "minor": 0.70, "patch": 0.20}
        else:
            probs = {"major": 0.05, "minor": 0.20, "patch": 0.75}
        return {"pred": max(probs, key=probs.get), "probs": probs}


# ---------- factory ---------------------------------------------------------

PROVIDERS = {
    "typesafe":  TypeSafeClient,
    "openai":    OpenAIClient,
    "anthropic": AnthropicClient,
    "ollama":    OllamaClient,
    "mock":      MockClient,
}


def make_client(provider: str, model: str, cache_path: str, **kw) -> LLMClient:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; "
                         f"choose one of {sorted(PROVIDERS)}")
    return PROVIDERS[provider](model=model, cache_path=cache_path, **kw)
