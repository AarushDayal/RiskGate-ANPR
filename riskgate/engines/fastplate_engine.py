"""Small local ONNX plate reader used for a fair OCR comparison."""

import time

import numpy as np
from fast_plate_ocr import LicensePlateRecognizer

from riskgate import config
from riskgate.contracts import EngineResult


class FastPlateEngine:
    def __init__(self) -> None:
        folder = config.ROOT / "weights/fastplate"
        model = folder / "cct_s_v2_global.onnx"
        plate_config = folder / "cct_s_v2_global_plate_config.yaml"
        if model.exists() and plate_config.exists():
            self.reader = LicensePlateRecognizer(onnx_model_path=model, plate_config_path=plate_config, device="cpu")
        else:
            self.reader = LicensePlateRecognizer("cct-s-v2-global-model", device="cpu")

    def read(self, image: np.ndarray) -> EngineResult:
        try:
            start = time.perf_counter()
            predictions = self.reader.run(image, return_confidence=True)
            latency = (time.perf_counter() - start) * 1000
            if not predictions:
                return EngineResult("", 0.0, latency, error="no text detected")
            prediction = predictions[0]
            confidence = float(np.mean(prediction.char_probs)) if len(prediction.char_probs) else 0.0
            return EngineResult(prediction.plate, confidence, latency, boxes_found=1)
        except (RuntimeError, ValueError, OSError) as error:
            return EngineResult("", 0.0, 0.0, error=str(error)[:200])
