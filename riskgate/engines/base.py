from typing import Protocol

import numpy as np

from riskgate.contracts import EngineResult


class Engine(Protocol):
    """Anything with read(image) -> EngineResult; must never raise."""

    def read(self, image: np.ndarray) -> EngineResult: ...
