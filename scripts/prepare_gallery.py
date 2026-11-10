"""Copy twelve manually transcribed demo photos from the box-only source."""

import csv
import shutil

import cv2

from riskgate import config

SOURCE = config.ROOT / "data/extra_detection/images"
GALLERY = config.ROOT / "data/sample_gallery"
PLATES = {
    "00000139.jpg": "TN76AZ1197",
    "00000228.jpg": "AP37DS3777",
    "00000165.jpg": "KL39E1895",
    "00000089.jpg": "UP65CW9680",
    "00000126.jpg": "MH01AE8017",
    "00000006.jpg": "KA41EN1051",
    "00000241.jpg": "AP13AA8705",
    "00000199.jpg": "MH20DY2366",
    "00000015.jpg": "JH24C1100",
    "00000004.jpg": "KA41EM7969",
    "00000381.jpg": "KA02JG1061",
    "00000203.jpg": "KA02MK6564",
}


def main() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, truth in PLATES.items():
        source = SOURCE / name
        if not source.exists():
            raise FileNotFoundError(source)
        target = GALLERY / name
        shutil.copy2(source, target)
        image = cv2.imread(str(source))
        height, width = image.shape[:2]
        label = (config.ROOT / "data/extra_detection/labels" / f"{source.stem}.txt").read_text().splitlines()[0].split()
        cx, cy, bw, bh = map(float, label[1:5])
        x1, y1 = max(0, int((cx - bw/2) * width)), max(0, int((cy - bh/2) * height))
        x2, y2 = min(width, int((cx + bw/2) * width)), min(height, int((cy + bh/2) * height))
        crop_path = GALLERY / "crops" / name
        crop_path.parent.mkdir(exist_ok=True)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid plate box for {name}")
        cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])
        rows.append({"image_path": str(target.relative_to(config.ROOT)),
                     "crop_path": str(crop_path.relative_to(config.ROOT)), "truth": truth,
                     "label_source": "manual_visual_transcription",
                     "source": "kedarsai/indian-license-plates-with-labels"})
    with (GALLERY / "manifest.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} labeled demo photos in {GALLERY}")


if __name__ == "__main__":
    main()
