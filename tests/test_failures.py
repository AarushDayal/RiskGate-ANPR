import numpy as np

from riskgate.cascade import process
from riskgate.contracts import EngineResult
from riskgate.engines.gemini_engine import GeminiEngine
from riskgate.engines.mock import MockEngine
from riskgate.features import FEATURE_NAMES


MODEL = {"feature_names": FEATURE_NAMES, "coefficients": [0] * len(FEATURE_NAMES),
         "intercept": 0, "mean": [0] * len(FEATURE_NAMES), "scale": [1] * len(FEATURE_NAMES),
         "thresholds": {"0.05": {"accept_at": None, "review_below": 0.3}}}
CROP = np.zeros((20, 60, 3), dtype=np.uint8)


class FailedGemini:
    calls_used = 0

    def __init__(self, message):
        self.message = message

    def read(self, crop):
        return EngineResult("", 0, 0, error=self.message)


def test_low_quality_and_empty_read():
    assert process(CROP[:5], MockEngine(), FailedGemini("timeout"), MODEL).route == "REVIEW"
    assert process(CROP, MockEngine(text=""), FailedGemini("timeout"), MODEL).route == "REVIEW"


def test_second_reader_failures():
    for error in ("timeout", "malformed JSON", "quota exceeded"):
        decision = process(CROP, MockEngine(text="KA01AB1234"), FailedGemini(error), MODEL)
        assert decision.route == "REVIEW" and decision.gemini_calls_used == 0


def test_dry_run_never_calls_and_cache_replays(tmp_path):
    engine = GeminiEngine(cache_dir=tmp_path)
    assert engine.read(CROP).error == "dry run"
    assert engine.would_call == 1 and engine.calls_used == 0
    from riskgate.cache import key
    from riskgate.engines.gemini_engine import PROMPT_VERSION
    import cv2
    digest = key(cv2.imencode(".png", CROP)[1].tobytes(), engine.model, PROMPT_VERSION)
    engine.cache.put(digest, {"text": "KA01AB1234"})
    assert engine.read(CROP).text == "KA01AB1234"
    assert engine.calls_used == 0


def test_malformed_json_is_cached_and_quota_is_hard(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MAX_CALLS", "1")
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return b'{"candidates":[{"content":{"parts":[{"text":"not-json"}]}}]}'

    def fake_urlopen(req, timeout):
        calls.append(1)
        return Response()

    monkeypatch.setattr("riskgate.engines.gemini_engine.request.urlopen", fake_urlopen)
    engine = GeminiEngine(run=True, cache_dir=tmp_path)
    assert engine.read(CROP).error.startswith("Gemini failed")
    assert engine.read(CROP).error.startswith("Gemini failed")
    assert len(calls) == 1 and engine.calls_used == 1
    assert engine.read(np.ones_like(CROP)).error == "Gemini call limit reached"
