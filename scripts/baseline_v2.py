"""Compare two local readers on identical crops, then freeze one on train data."""

import csv
import json

import cv2

from riskgate import config
from riskgate.data import crop, load_all_samples, make_splits_v2
from riskgate.engines.easyocr_engine import EasyOCREngine
from riskgate.engines.fastplate_engine import FastPlateEngine
from riskgate.features import plate_features
from riskgate.metrics import cer, exact_match, normalize
from riskgate.router import wilson_interval


def write_csv(path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    samples = load_all_samples()
    splits = make_splits_v2(samples)
    owner = {group: split for split, groups in splits.items() for group in groups}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    readers = {"EasyOCR": EasyOCREngine(config.READER_UPSCALE, config.READER_ALLOWLIST),
               "FastPlateOCR": FastPlateEngine()}
    rows = []
    for index, sample in enumerate(samples, 1):
        image = cv2.imread(sample.image_path)
        if image is None:
            continue
        plate = image if sample.source == "kaggle" else crop(image, sample.bbox)
        if plate is None:
            continue
        for name, reader in readers.items():
            result = reader.read(plate)
            pred = normalize(result.text)
            rows.append({"image": sample.image_path, "group_id": sample.group_id,
                         "source": sample.source, "split": owner[sample.group_id],
                         "reader": name, "truth": normalize(sample.truth), "pred": pred,
                         "ok": int(exact_match(pred, sample.truth)), "cer": cer(pred, sample.truth),
                         "latency_ms": result.latency_ms, **plate_features(plate, result)})
        if index % 100 == 0:
            print(f"Read {index}/{len(samples)} plates", flush=True)
    write_csv(config.RESULTS_DIR / "ocr_comparison.csv", rows)
    train = [row for row in rows if row["split"] == "train"]
    chosen = min(readers, key=lambda name: (-sum(row["ok"] for row in train if row["reader"] == name),
                                             sum(row["cer"] for row in train if row["reader"] == name)))
    selected = [row for row in rows if row["reader"] == chosen]
    write_csv(config.RESULTS_DIR / "baseline_v2.csv", selected)
    (config.RESULTS_DIR / "selected_reader.json").write_text(json.dumps({"reader": chosen}, indent=2) + "\n")
    lines = [f"Chosen on train split: {chosen}",
             "Reader | Split | Source | n | Exact | 95% Wilson interval | Mean CER | Mean latency ms",
             "--- | --- | --- | ---: | ---: | --- | ---: | ---:"]
    for name in readers:
        for split in splits:
            subset = [row for row in rows if row["reader"] == name and row["split"] == split]
            if not subset:
                continue
            n, good = len(subset), sum(row["ok"] for row in subset)
            lo, hi = wilson_interval(n - good, n)
            lines.append(f"{name} | {split} | {subset[0]['source']} | {n} | {good/n:.3f} | "
                         f"[{1-hi:.3f}, {1-lo:.3f}] | {sum(row['cer'] for row in subset)/n:.3f} | "
                         f"{sum(row['latency_ms'] for row in subset)/n:.1f}")
    (config.RESULTS_DIR / "baseline_v2_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
