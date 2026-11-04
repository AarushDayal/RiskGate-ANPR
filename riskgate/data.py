import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from riskgate import config
from riskgate.contracts import PlateSample


def load_samples(data_dir: Path = config.DATA_DIR) -> list[PlateSample]:
    """Parse every XML into PlateSamples: skip objects with no plate text, clip bbox to image."""
    samples = []
    for xml_path in sorted((data_dir / "Annotations").glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        name = root.findtext("filename")
        img = cv2.imread(str(data_dir / "images" / name))
        if img is None:
            continue
        H, W = img.shape[:2]
        for obj in root.findall("object"):
            truth = next(
                (
                    a.findtext("value")
                    for a in obj.iter("attribute")
                    if a.findtext("name") == "number_plate_text"
                ),
                None,
            )
            if not truth:
                continue
            bb = obj.find("bndbox")
            x1, y1, x2, y2 = (
                int(float(bb.findtext(c))) for c in ("xmin", "ymin", "xmax", "ymax")
            )
            samples.append(
                PlateSample(
                    image_path=str(data_dir / "images" / name),
                    bbox=(max(0, x1), max(0, y1), min(W, x2), min(H, y2)),
                    truth=truth,
                )
            )
    return samples


def make_splits(samples: list[PlateSample]) -> dict[str, list[str]]:
    """Split ALL dataset image NAMES (never plates of one image apart) into calib/val/test.

    # ponytail: split all 47 images, not just the ~21 that carry labelled plates -
    # a Part-2 split should cover every image the eval might see later.
    """
    names = sorted(p.name for p in config.ANNOTATIONS_DIR.glob("*.xml"))
    rng = random.Random(config.SEED)
    rng.shuffle(names)
    n_calib = round(config.SPLIT_FRACTIONS[0] * len(names))
    n_val = round(config.SPLIT_FRACTIONS[1] * len(names))
    splits = {
        "calib": names[:n_calib],
        "val": names[n_calib : n_calib + n_val],
        "test": names[n_calib + n_val :],
    }
    assert not set(splits["calib"]) & set(splits["val"]) and not set(
        splits["calib"]
    ) & set(splits["test"]) and not set(splits["val"]) & set(splits["test"]), \
        "an image appeared in two splits"
    config.SPLITS_JSON.write_text(json.dumps(splits, indent=2))
    return splits


def crop(img, bbox):
    """Crop bbox out of image, honouring config.MIN_CROP_H (empty crop -> None)."""
    x1, y1, x2, y2 = bbox
    c = img[y1:y2, x1:x2]
    return None if c.size == 0 or c.shape[0] < config.MIN_CROP_H else c
