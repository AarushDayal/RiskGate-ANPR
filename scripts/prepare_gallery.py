"""Copy visually checked full vehicle photos for the noncommercial demo gallery."""

import csv
import shutil
from pathlib import Path

from riskgate import config

GALLERY = config.ROOT / "data/sample_gallery"
SOURCE = "saisirishan/indian-vehicle-dataset"

# Every source file is copied unmodified. Crops are made in memory by the app.
SCENES = (
    "google_images/0b24d6ed-d32f-420d-bc18-03deece29073___2015-Maruti-Ciaz-Test-Drive-Review.jpg.jpeg",
    "google_images/2c9306ab-3454-4ca0-89fd-db3f51dabcef___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-front-three-quarters-on-plain-ground.jpg",
    "google_images/car-wbs-MH14EH7958_00000.jpeg",
    "google_images/car-wbs-HR26CT6702_00000.jpeg",
    "google_images/car-wbs-KA03NA8385_00000.png",
    "google_images/car-wbs-MH03BS7778_00001.jpeg",
    "google_images/2424211e-e6ba-4478-8ed1-6a04e25499c7___main-qimg-7673aeac86d0ef987deb837552c5dd4c-1.png",
    "video_images/video9_360.jpg",
    "video_images/video11_1150.jpg",
    "video_images/video10_870.jpg",
    "video_images/car-wbs-MH01DE2780_00000.png",
    "google_images/e354039f-d73c-4f12-9df5-f0129345fa79___scorpio.JPG.jpeg",
)


def main() -> None:
    with (config.ROOT / "data/indian_manifest.csv").open(newline="") as source:
        labels = {row["image_path"]: row for row in csv.DictReader(source)}
    GALLERY.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, relative in enumerate(SCENES, 1):
        source_path = config.ROOT / "data/indian" / relative
        row = labels[str(source_path.relative_to(config.ROOT))]
        target = GALLERY / f"scene_{index:02d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, target)
        rows.append({"image_path": str(target.relative_to(config.ROOT)), "truth": row["truth"],
                     "xmin": row["xmin"], "ymin": row["ymin"],
                     "xmax": row["xmax"], "ymax": row["ymax"],
                     "label_source": "xml_text_visually_checked", "source": SOURCE,
                     "license": "CC BY-NC-ND 4.0"})
    with (GALLERY / "manifest.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} full vehicle scenes in {GALLERY}")


if __name__ == "__main__":
    main()
