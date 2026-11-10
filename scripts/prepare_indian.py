"""Create crops and a reproducible manifest from the downloaded image/XML pairs."""

import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from riskgate import config
from riskgate.metrics import normalize

SOURCE = config.ROOT / "data/indian"
XML_LABELS = config.ROOT / "data/labels/kaggle_xml_labels.csv"
DRAFTS = config.ROOT / "draft_labels.csv"
MANIFEST = config.ROOT / "data/indian_manifest.csv"
CROPS = config.ROOT / "data/crops/indian"


def video_key(path: Path) -> str | None:
    if path.parts[0] != "video_images":
        return None
    stem = path.stem
    if re.match(r"video\d+_\d+$", stem):
        return stem.split("_")[0]
    if re.search(r"_\d{5}$", stem):
        return stem.rsplit("_", 1)[0]
    return stem


def records() -> list[dict]:
    rows = []
    for xml_path in sorted(SOURCE.rglob("*.xml")):
        rel = xml_path.relative_to(SOURCE)
        root = ET.parse(xml_path).getroot()
        image_path = xml_path.parent / (root.findtext("filename") or "")
        if not image_path.is_file():
            print(f"Skipping missing image for {rel}")
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for index, obj in enumerate(root.findall("object")):
            truth = normalize(obj.findtext("name") or "")
            box = obj.find("bndbox")
            if not truth or box is None:
                continue
            x1, y1, x2, y2 = [int(float(box.findtext(key) or 0)) for key in ("xmin", "ymin", "xmax", "ymax")]
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_path = CROPS / f"{rel.with_suffix('').as_posix().replace('/', '__')}_{index}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])
            rows.append({"image_path": str(image_path.relative_to(config.ROOT)),
                         "crop_path": str(crop_path.relative_to(config.ROOT)),
                         "truth": truth, "source_detail": rel.parts[0],
                         "video_key": video_key(rel), "xmin": x1, "ymin": y1,
                         "xmax": x2, "ymax": y2})
    return rows


def assign_groups(rows: list[dict]) -> None:
    """Connect samples sharing a plate text or a source video."""
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        keys = ["plate:" + row["truth"]]
        if row["video_key"]:
            keys.append("video:" + row["video_key"])
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    for i, row in enumerate(rows):
        row["group_id"] = f"g{root(i):04d}"


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)


def main() -> None:
    rows = records()
    if not rows:
        raise ValueError(f"No usable image/XML pairs found in {SOURCE}")
    assign_groups(rows)
    write_csv(MANIFEST, rows, list(rows[0]))
    write_csv(XML_LABELS, rows, ["crop_path", "group_id", "truth"])
    write_csv(DRAFTS, [{**row, "draft": row["truth"]} for row in rows],
              ["crop_path", "group_id", "draft"])
    print(json.dumps({"images": len({row['image_path'] for row in rows}),
                      "plates": len(rows), "groups": len({row['group_id'] for row in rows}),
                      "source_counts": {source: sum(row["source_detail"] == source for row in rows)
                                        for source in sorted({row["source_detail"] for row in rows})}}, indent=2))
    print("Labels came from XML annotations and still need visual auditing.")


if __name__ == "__main__":
    main()
