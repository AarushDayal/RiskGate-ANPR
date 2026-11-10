"""Add public box-only images to detector training while protecting existing holdouts."""

import csv
import json
from pathlib import Path

import cv2

from riskgate import config

SOURCE = config.ROOT / "data/extra_detection"
OUTPUT = config.ROOT / "data/extra_detection_import.csv"


def dhash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    small = cv2.resize(image, (9, 8))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def valid_labels(path: Path) -> bool:
    try:
        lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
        return bool(lines) and all(len(line) == 5 and line[0] == "0" and
                                   all(0 <= float(value) <= 1 for value in line[1:]) and
                                   float(line[3]) > 0 and float(line[4]) > 0 for line in lines)
    except (OSError, ValueError):
        return False


def main() -> None:
    splits = json.loads(config.SPLITS_V2.read_text())
    protected = set().union(*(set(splits[name]) for name in ("calib", "val", "test")))
    with (config.ROOT / "data/indian_manifest.csv").open(newline="") as source:
        holdout_paths = {config.ROOT / row["image_path"] for row in csv.DictReader(source)
                         if "kaggle:" + row["group_id"] in protected}
    holdout_hashes = [dhash(path) for path in sorted(holdout_paths)]
    images = sorted((SOURCE / "images").glob("*.jpg"))
    image_dir = config.ROOT / "data/yolo/images/train"
    label_dir = config.ROOT / "data/yolo/labels/train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in images:
        label = SOURCE / "labels" / f"{path.stem}.txt"
        status = "included"
        if not valid_labels(label):
            status = "missing_or_invalid_label"
        else:
            fingerprint = dhash(path)
            if any((fingerprint ^ other).bit_count() <= 3 for other in holdout_hashes):
                status = "possible_holdout_duplicate"
        if status == "included":
            link = image_dir / f"extra_{path.stem}.jpg"
            if not link.exists():
                link.symlink_to(path)
            (label_dir / f"extra_{path.stem}.txt").write_text(label.read_text())
        rows.append({"image_path": str(path.relative_to(config.ROOT)), "status": status})
    with OUTPUT.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["image_path", "status"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})}
    print(json.dumps({"total_full_images": len(images), "status": counts,
                      "unlabeled_plate_crops": len(list((SOURCE / "images").glob("*.png")))}, indent=2))


if __name__ == "__main__":
    main()
