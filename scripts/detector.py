"""Train and evaluate a small full-image plate detector on the grouped split."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

from riskgate import config

MANIFEST = config.ROOT / "data/indian_manifest.csv"
DATASET = config.ROOT / "data/yolo"
WEIGHTS = config.ROOT / "weights/plate_detector.pt"


def load_rows() -> list[dict]:
    with MANIFEST.open(newline="") as source:
        return list(csv.DictReader(source))


def box_to_yolo(row: dict, width: int, height: int) -> str:
    x1, y1, x2, y2 = (float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
    return f"0 {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}\n"


def prepare() -> dict:
    splits = json.loads((config.ROOT / "splits_v2.json").read_text())
    owner = {group: split for split, groups in splits.items() for group in groups}
    counts = {split: 0 for split in ("train", "val", "test")}
    for index, row in enumerate(load_rows()):
        split = owner["kaggle:" + row["group_id"]]
        if split not in counts:
            continue
        image_path = config.ROOT / row["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        name = f"plate_{index:04d}"
        link = image_dir / f"{name}{image_path.suffix.lower()}"
        if not link.exists():
            link.symlink_to(image_path)
        (label_dir / f"{name}.txt").write_text(box_to_yolo(row, width, height))
        counts[split] += 1
    (DATASET / "dataset.yaml").write_text(
        f"path: {DATASET}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: license_plate\n"
    )
    return counts


def overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def evaluate(model: YOLO) -> dict:
    splits = json.loads((config.ROOT / "splits_v2.json").read_text())
    test = set(splits["test"])
    rows = [row for row in load_rows() if "kaggle:" + row["group_id"] in test]
    found, correct, predictions = 0, 0, 0
    details = []
    for row in rows:
        result = model.predict(str(config.ROOT / row["image_path"]), conf=0.25, verbose=False)[0]
        boxes = [tuple(float(v) for v in box.xyxy[0].tolist()) for box in result.boxes]
        predictions += len(boxes)
        truth = tuple(float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
        best = max((overlap(box, truth) for box in boxes), default=0.0)
        found += bool(boxes)
        correct += best >= 0.5
        details.append({"image": row["image_path"], "truth": row["truth"],
                        "found": bool(boxes), "matched_iou_05": best >= 0.5, "best_iou": round(best, 3)})
    report = {"test_images": len(rows), "boxes_predicted": predictions,
              "images_with_detection": found, "matched_at_iou_05": correct,
              "recall_at_iou_05": correct / len(rows),
              "precision_at_iou_05": correct / predictions if predictions else 0.0}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "detector_test.json").write_text(json.dumps(report, indent=2) + "\n")
    with (config.RESULTS_DIR / "detector_test_details.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--start-from", default="yolo11n.pt")
    parser.add_argument("--name", default="detector_training")
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(), indent=2))
    elif args.mode == "train":
        if not (DATASET / "dataset.yaml").exists():
            prepare()
        model = YOLO(args.start_from)
        model.train(data=str(DATASET / "dataset.yaml"), epochs=args.epochs, imgsz=640,
                    batch=8, device=0, project=str(config.RESULTS_DIR), name=args.name,
                    exist_ok=True, workers=2, seed=config.SEED)
        WEIGHTS.parent.mkdir(exist_ok=True)
        shutil.copyfile(config.RESULTS_DIR / args.name / "weights/best.pt", WEIGHTS)
        print(json.dumps(evaluate(YOLO(str(WEIGHTS))), indent=2))
    else:
        print(json.dumps(evaluate(YOLO(str(WEIGHTS))), indent=2))


if __name__ == "__main__":
    main()
