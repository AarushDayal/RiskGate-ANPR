from riskgate.contracts import EngineResult


class MockEngine:
    """Deterministic engine for tests: no model, no network, no GPU."""

    def __init__(self, text: str = "KL34A465", conf: float = 0.9, error: str | None = None):
        self.text, self.conf, self.error = text, conf, error

    def read(self, image) -> EngineResult:
        return EngineResult("" if self.error else self.text, self.conf, 0.0, self.error)
