"""Selective OCR followed by one cached second opinion when needed."""

from dataclasses import dataclass
from time import perf_counter

from riskgate.calibration import predict
from riskgate.grammar import classify
from riskgate.metrics import normalize
from riskgate.router import route


@dataclass
class Decision:
    final_text: str
    route: str
    confidence: float
    reason: str
    engines_used: list[str]
    latency_ms: float
    gemini_calls_used: int


def process(crop, easyocr, gemini, model: dict, target: str = "0.05") -> Decision:
    from riskgate.features import plate_features

    started = perf_counter()
    if crop is None or crop.size == 0 or crop.shape[0] < 12:
        return Decision("", "REVIEW", 0, "crop too small or empty", [], 0, 0)
    first = easyocr.read(crop)
    used = ["EasyOCR"]
    if first.error or not normalize(first.text):
        return Decision("", "REVIEW", 0, first.error or "empty EasyOCR read", used,
                        (perf_counter() - started) * 1000, 0)
    conf = predict(model, plate_features(crop, first))
    initial = route(first.text, conf, model["thresholds"][target])
    if initial == "ACCEPT":
        return Decision(normalize(first.text), "ACCEPT", conf, "calibrated threshold met", used,
                        (perf_counter() - started) * 1000, 0)
    if initial == "REVIEW":
        return Decision(normalize(first.text), "REVIEW", conf, "low confidence", used,
                        (perf_counter() - started) * 1000, 0)
    before = gemini.calls_used
    second = gemini.read(crop)
    used.append("Gemini")
    calls = gemini.calls_used - before
    agree = not second.error and normalize(second.text) == normalize(first.text)
    # INVALID grammar remains in review even if two engines echo the same junk.
    if agree and classify(first.text) != "INVALID":
        return Decision(normalize(first.text), "ACCEPT", conf, "two engines agree", used,
                        (perf_counter() - started) * 1000, calls)
    return Decision(normalize(first.text), "REVIEW", conf,
                    second.error or "engines disagree or plate format invalid", used,
                    (perf_counter() - started) * 1000, calls)
