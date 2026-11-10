"""Stable numeric signals for the correctness calibrator."""

import numpy as np

from riskgate.contracts import EngineResult
from riskgate.grammar import classify


FEATURE_NAMES = ("raw_conf", "grammar", "text_length", "crop_height", "crop_width", "boxes_found")


def plate_features(crop: np.ndarray, result: EngineResult) -> dict[str, float]:
    height, width = crop.shape[:2]
    return {"raw_conf": float(result.raw_conf),
            "grammar": float({"INVALID": 0, "NONSTANDARD": 1, "VALID": 2}[classify(result.text)]),
            "text_length": float(len(result.text)), "crop_height": float(height),
            "crop_width": float(width), "boxes_found": float(result.boxes_found)}
