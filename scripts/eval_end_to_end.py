"""One-time full-image evaluation: detector miss counts as an OCR failure."""

import argparse
import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from riskgate import config
from riskgate.calibration import load, predict
from riskgate.engines.easyocr_engine import EasyOCREngine
from riskgate.engines.fastplate_engine import FastPlateEngine
from riskgate.features import plate_features
from riskgate.metrics import cer, exact_match, normalize
from riskgate.router import route, wilson_interval

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "end_to_end_test.json")
    parser.add_argument("--weights", type=Path, default=config.ROOT / "weights/plate_detector.pt")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; keep the first evaluation run")
    splits = json.loads((config.ROOT / "splits_v2.json").read_text())
    test = set(splits["test"])
    with (config.ROOT / "data/indian_manifest.csv").open(newline="") as source:
        samples = [row for row in csv.DictReader(source) if "kaggle:" + row["group_id"] in test]
    name = json.loads((config.RESULTS_DIR / "selected_reader.json").read_text())["reader"]
    reader = FastPlateEngine() if name == "FastPlateOCR" else EasyOCREngine(config.READER_UPSCALE, config.READER_ALLOWLIST)
    detector = YOLO(str(args.weights))
    calibration = load(config.ROOT / "calibration.json")
    details = []
    for row in samples:
        image = cv2.imread(str(config.ROOT / row["image_path"]))
        if image is None:
            continue
        result = detector.predict(image, conf=0.25, verbose=False)[0]
        pred, confidence, decision, read_time = "", 0.0, "REVIEW", 0.0
        if len(result.boxes):
            box = max(result.boxes, key=lambda item: float(item.conf[0]))
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            plate = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
            if plate.size:
                read = reader.read(plate)
                pred = normalize(read.text)
                read_time = read.latency_ms
                if pred:
                    confidence = predict(calibration, plate_features(plate, read))
                    decision = route(pred, confidence, calibration["thresholds"]["0.10"])
        details.append({"image": row["image_path"], "truth": row["truth"], "pred": pred,
                        "ok": int(exact_match(pred, row["truth"])), "cer": cer(pred, row["truth"]),
                        "detected": int(bool(len(result.boxes))), "confidence": confidence,
                        "route": decision, "ocr_latency_ms": read_time})
    n = len(details)
    good = sum(row["ok"] for row in details)
    accepted = [row for row in details if row["route"] == "ACCEPT"]
    errors = sum(1 - row["ok"] for row in accepted)
    report = {"reader": name, "n": n, "detected": sum(row["detected"] for row in details),
              "whole_plate_exact": good, "whole_plate_accuracy": good/n,
              "accuracy_95ci": tuple(1-x for x in reversed(wilson_interval(n-good, n))),
              "mean_cer": sum(row["cer"] for row in details)/n,
              "accepted": len(accepted), "accepted_errors": errors,
              "accepted_error": errors/len(accepted) if accepted else None,
              "coverage": len(accepted)/n}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_name(args.output.stem + "_details.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
