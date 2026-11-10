from riskgate.calibration import fit, predict
from riskgate.features import FEATURE_NAMES
from riskgate.router import choose_thresholds, route, wilson_interval


def test_calibrator_roundtrip():
    rows = [{**dict.fromkeys(FEATURE_NAMES, float(i % 5)), "ok": int(i % 2), "image": str(i)}
            for i in range(30)]
    model = fit(rows)
    assert set(model["feature_names"]) == set(FEATURE_NAMES)
    assert 0 <= predict(model, rows[0]) <= 1


def test_wilson_and_invalid_rule():
    assert wilson_interval(0, 100)[1] < 0.05
    rows = [{"pred": "KA01AB1234", "calibrated_conf": i / 100, "ok": 1}
            for i in range(100)]
    thresholds = choose_thresholds(rows)
    assert thresholds["0.05"]["accept_at"] is not None
    assert route("junk", 1.0, thresholds["0.05"]) != "ACCEPT"
