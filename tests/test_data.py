from riskgate.contracts import EngineResult
from riskgate.data import crop, load_samples, make_splits
from riskgate.engines.mock import MockEngine


def test_load_samples_real_data():
    samples = load_samples()
    assert len(samples) >= 20  # ~25 labelled plates
    for s in samples:
        assert s.truth.strip()
        x1, y1, x2, y2 = s.bbox
        assert x2 > x1 and y2 > y1


def test_splits_disjoint():
    samples = load_samples()
    sp = make_splits(samples)
    assert len(sp["calib"]) == 19 and len(sp["val"]) == 14 and len(sp["test"]) == 14
    all_img = sp["calib"] + sp["val"] + sp["test"]
    assert len(all_img) == len(set(all_img)) == 47
    import json

    assert json.loads(open("splits.json").read())["test"] == sp["test"]


def test_crop_and_engine():
    import cv2

    samples = load_samples()
    img = cv2.imread(samples[0].image_path)
    c = crop(img, samples[0].bbox)
    assert c is not None and c.size
    # engine failure path: error result, no raise, no fake text
    r = MockEngine(error="boom").read(img)
    assert isinstance(r, EngineResult) and r.error == "boom" and r.text == ""
    # happy path
    r = MockEngine().read(img)
    assert r.text == "KL34A465" and r.error is None
