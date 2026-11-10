"""Read reviewed plates and write the frozen-reader baseline with features."""

import csv
from pathlib import Path

import cv2

from riskgate import config
from riskgate.data import crop, load_all_samples, make_splits_v2
from riskgate.engines.easyocr_engine import EasyOCREngine
from riskgate.features import plate_features
from riskgate.metrics import cer, exact_match, normalize
from riskgate.router import wilson_interval


def main() -> None:
    samples = load_all_samples()
    splits = make_splits_v2(samples)
    owner = {group: split for split, groups in splits.items() for group in groups}
    reader = EasyOCREngine(upscale=config.READER_UPSCALE, allowlist=config.READER_ALLOWLIST)
    rows = []
    for sample in samples:
        img = cv2.imread(sample.image_path)
        if img is None:
            continue
        plate = img if sample.source == "kaggle" else crop(img, sample.bbox)
        if plate is None:
            continue
        result = reader.read(plate)
        pred = normalize(result.text)
        rows.append({"image": sample.image_path, "group_id": sample.group_id,
                     "source": sample.source, "split": owner[sample.group_id],
                     "truth": normalize(sample.truth), "pred": pred,
                     "ok": int(exact_match(pred, sample.truth)), "cer": cer(pred, sample.truth),
                     "latency_ms": result.latency_ms,
                     **plate_features(plate, result)})
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with (config.RESULTS_DIR / "baseline_v2.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Frozen reader baseline", "", f"Config: allowlist={config.READER_ALLOWLIST}, upscale={config.READER_UPSCALE}",
             "", "| Split | Source | n | Exact match | 95% Wilson interval | CER | Confidence correct | Confidence wrong |",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for split in splits:
        for source in ("kaggle", "datacluster"):
            subset = [r for r in rows if r["split"] == split and r["source"] == source]
            if not subset:
                continue
            n, good = len(subset), sum(r["ok"] for r in subset)
            lo, hi = wilson_interval(n - good, n)
            means = [sum(r["raw_conf"] for r in subset if r["ok"] == ok) /
                     max(1, sum(r["ok"] == ok for r in subset)) for ok in (1, 0)]
            lines.append(f"| {split} | {source} | {n} | {good/n:.3f} | [{1-hi:.3f}, {1-lo:.3f}] | "
                         f"{sum(r['cer'] for r in subset)/n:.3f} | {means[0]:.3f} | {means[1]:.3f} |")
    (config.RESULTS_DIR / "baseline_v2_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
