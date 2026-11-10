from dataclasses import dataclass


@dataclass
class PlateSample:
    """One labelled plate: source image, pixel bbox (x1, y1, x2, y2), ground-truth text."""

    image_path: str
    bbox: tuple[int, int, int, int]
    truth: str
    source: str = "datacluster"
    group_id: str = ""


@dataclass
class EngineResult:
    """One engine read. On failure error is set and text stays "" - never raise, never fake."""

    text: str
    raw_conf: float
    latency_ms: float
    error: str | None = None
    boxes_found: int = 0
