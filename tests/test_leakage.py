from pathlib import Path

from riskgate.contracts import PlateSample
from riskgate.data import make_splits_v2


def test_group_split_keeps_sources_and_images_apart(tmp_path: Path):
    samples = [PlateSample(f"{i}_{j}.jpg", (0, 0, 1, 1), "AB12", "kaggle", f"kaggle:{i}")
               for i in range(10) for j in range(2)]
    samples.append(PlateSample("d.jpg", (0, 0, 1, 1), "AB12", "datacluster", "datacluster:d"))
    splits = make_splits_v2(samples, tmp_path / "splits_v2.json")
    assert len(set().union(*(set(v) for v in splits.values()))) == sum(map(len, splits.values()))
    assert all(splits[name] for name in ("train", "calib", "val", "test"))
    assert splits["cross_dataset_test"] == ["datacluster:d"]
