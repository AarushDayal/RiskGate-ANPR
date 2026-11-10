"""Capped Gemini second opinion. Default mode never makes a network call."""

import base64
import json
import os
import time
from pathlib import Path
from urllib import error, request

import cv2

from riskgate import config
from riskgate.cache import Cache, key
from riskgate.contracts import EngineResult
from riskgate.metrics import normalize


PROMPT_VERSION = "plate-json-v1"
PROMPT = "Read this Indian vehicle number plate crop. Return only JSON: {\"plate\":\"TEXT\"}. If unreadable, use an empty string."


class GeminiEngine:
    def __init__(self, run: bool = False, offline: bool = False, model: str = config.GEMINI_MODEL,
                 cache_dir: Path = config.RESULTS_DIR / "gemini_cache"):
        self.run, self.offline, self.model = run, offline, model
        self.cache = Cache(cache_dir)
        self.max_calls = max(0, int(os.getenv("GEMINI_MAX_CALLS", "0")))
        self.calls_used = 0
        self.would_call = 0

    def read(self, crop) -> EngineResult:
        start = time.perf_counter()
        try:
            ok, encoded = cv2.imencode(".png", crop)
        except (cv2.error, TypeError, ValueError):
            ok = False
        if not ok:
            return EngineResult("", 0, 0, error="could not encode crop")
        digest = key(encoded.tobytes(), self.model, PROMPT_VERSION)
        try:
            cached = self.cache.get(digest)
        except (OSError, ValueError):
            return EngineResult("", 0, 0, error="Gemini cache unreadable")
        if cached is not None:
            return EngineResult(cached.get("text", ""), 1.0 if cached.get("text") else 0.0,
                                (time.perf_counter() - start) * 1000, cached.get("error"))
        if self.offline:
            return EngineResult("", 0, 0, error="offline cache miss")
        if not self.run:
            self.would_call += 1
            return EngineResult("", 0, 0, error="dry run")
        if self.calls_used >= self.max_calls:
            return EngineResult("", 0, 0, error="Gemini call limit reached")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return EngineResult("", 0, 0, error="GEMINI_API_KEY missing")
        self.calls_used += 1
        try:
            payload = {"contents": [{"parts": [{"text": PROMPT},
                       {"inline_data": {"mime_type": "image/png",
                                        "data": base64.b64encode(encoded.tobytes()).decode()}}]}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
            req = request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key}, method="POST")
            with request.urlopen(req, timeout=20) as response:
                body = json.load(response)
            answer = json.loads(body["candidates"][0]["content"]["parts"][0]["text"])
            raw = answer["plate"]
            if not isinstance(raw, str) or not normalize(raw):
                raise ValueError("empty or invalid Gemini plate")
            value = {"text": normalize(raw)}
        except (error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError,
                json.JSONDecodeError) as exc:
            value = {"text": "", "error": f"Gemini failed: {str(exc)[:150]}"}
        value["latency_ms"] = (time.perf_counter() - start) * 1000
        try:
            self.cache.put(digest, value)
        except OSError:
            return EngineResult("", 0, 0, error="Gemini cache write failed")
        return EngineResult(value["text"], 1.0 if value["text"] else 0.0,
                            (time.perf_counter() - start) * 1000, value.get("error"))
