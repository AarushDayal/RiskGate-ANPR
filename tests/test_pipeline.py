from pathlib import Path

from riskgate import config
from riskgate.data import load_all_samples
from scripts.detector import box_to_yolo, overlap
from scripts.import_extra_detection import valid_labels
from scripts.prepare_indian import assign_groups, video_key


def test_video_and_plate_groups_stay_together():
    assert video_key(Path("video_images/video11_1050.jpg")) == "video11"
    assert video_key(Path("video_images/car-wbs-MH01DE2780_00001.jpg")) == "car-wbs-MH01DE2780"
    assert video_key(Path("State-wise_OLX/MH/MH1.jpg")) is None
    rows = [{"truth": "AA1", "video_key": "video11"},
            {"truth": "BB2", "video_key": "video11"},
            {"truth": "BB2", "video_key": None},
            {"truth": "CC3", "video_key": None}]
    assign_groups(rows)
    assert rows[0]["group_id"] == rows[1]["group_id"] == rows[2]["group_id"]
    assert rows[3]["group_id"] != rows[0]["group_id"]


def test_detector_box_coordinates_and_overlap():
    row = {"xmin": 20, "ymin": 10, "xmax": 60, "ymax": 30}
    assert box_to_yolo(row, 100, 50) == "0 0.400000 0.400000 0.400000 0.400000\n"
    assert overlap((20, 10, 60, 30), (20, 10, 60, 30)) == 1
    assert overlap((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_human_review_overrides_xml_without_editing_it(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    labels = tmp_path / "data/labels"
    labels.mkdir(parents=True)
    (tmp_path / "crop.jpg").write_bytes(b"x")
    xml = labels / "kaggle_xml_labels.csv"
    xml.write_text("crop_path,group_id,truth\ncrop.jpg,g1,AB12\n")
    (labels / "kaggle_labels.csv").write_text("crop_path,group_id,truth\ncrop.jpg,g1,AB13\n")
    samples = load_all_samples(tmp_path / "empty")
    assert len(samples) == 1 and samples[0].truth == "AB13"
    assert "AB12" in xml.read_text()


def test_extra_detector_labels_reject_invalid_boxes(tmp_path):
    label = tmp_path / "plate.txt"
    label.write_text("0 0.5 0.5 0.4 0.2\n")
    assert valid_labels(label)
    label.write_text("0 0.5 0.5 1.4 0.2\n")
    assert not valid_labels(label)
