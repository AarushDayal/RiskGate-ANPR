import time

import cv2
import easyocr
import torch

from riskgate import config
from riskgate.contracts import EngineResult


def order_boxes(boxes: list) -> list:
    """Return EasyOCR results in reading order, grouping overlapping text rows."""
    rows = []
    for box in sorted(boxes, key=lambda item: sum(p[1] for p in item[0]) / 4):
        ys = [p[1] for p in box[0]]
        top, bottom = min(ys), max(ys)
        for row in rows:
            if min(bottom, row["bottom"]) - max(top, row["top"]) > 0.5 * min(bottom - top, row["bottom"] - row["top"]):
                row["boxes"].append(box)
                row["top"] = min(top, row["top"])
                row["bottom"] = max(bottom, row["bottom"])
                break
        else:
            rows.append({"top": top, "bottom": bottom, "boxes": [box]})
    return [box for row in sorted(rows, key=lambda r: r["top"])
            for box in sorted(row["boxes"], key=lambda item: min(p[0] for p in item[0]))]


class EasyOCREngine:
    """Wraps easyocr on GPU. Flags are config-level, default off (task 7)."""

    def __init__(self, upscale: bool = False, allowlist: bool = False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: expected cuda but got cpu - diagnose, do not silently continue")
        print(f"EasyOCREngine device: {device}")
        self.gpu_used = device == "cuda"
        self.upscale = upscale
        self.allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" if allowlist else None
        self.reader = easyocr.Reader(["en"], gpu=self.gpu_used)

    def read(self, image) -> EngineResult:
        try:
            if self.upscale:
                image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            t0 = time.perf_counter()
            res = self.reader.readtext(image, allowlist=self.allowlist)
            latency_ms = (time.perf_counter() - t0) * 1000
            if not res:
                return EngineResult("", 0.0, latency_ms, error="no text detected", boxes_found=0)
            if config.MIN_CROP_H and image.shape[0] < config.MIN_CROP_H:
                return EngineResult("", 0.0, latency_ms, error="crop too small")
            res = order_boxes(res)
            return EngineResult(
                "".join(r[1] for r in res),
                sum(r[2] for r in res) / len(res),
                latency_ms,
                boxes_found=len(res),
            )
        except (cv2.error, RuntimeError, ValueError, OSError) as e:
            return EngineResult("", 0.0, 0.0, error=str(e)[:200])
