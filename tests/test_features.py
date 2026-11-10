import numpy as np

from riskgate.contracts import EngineResult
from riskgate.features import plate_features


def test_features():
    f = plate_features(np.zeros((20, 60, 3), dtype=np.uint8),
                       EngineResult("KA01AB1234", 0.8, 1.0, boxes_found=2))
    assert f == {"raw_conf": 0.8, "grammar": 2.0, "text_length": 10.0,
                 "crop_height": 20.0, "crop_width": 60.0, "boxes_found": 2.0}
