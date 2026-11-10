"""Fit on calib, select on val, and perform a guarded one-time test."""

import argparse
import csv
import json
from pathlib import Path

from riskgate import config
from riskgate.calibration import data_hash, fit, load, predict, save
from riskgate.metrics import ece, risk_coverage
from riskgate.router import choose_thresholds, route, wilson_interval


READS = config.RESULTS_DIR / "baseline_v2.csv"
MODEL = config.CALIBRATION_MODEL
FINAL = config.RESULTS_DIR / "final_test.json"


def read_rows() -> list[dict]:
    with READS.open(newline="") as f:
        return list(csv.DictReader(f))


def fit_model(rows: list[dict]) -> None:
    calib = [r for r in rows if r["split"] == "calib"]
    val = [r for r in rows if r["split"] == "val"]
    if not val:
        raise ValueError("Validation split is empty")
    model = fit(calib)
    for r in val:
        r["calibrated_conf"] = predict(model, r)
    model["thresholds"] = choose_thresholds(val)
    model["validation_ece"] = {"raw": ece([float(r["raw_conf"]) for r in val],
                                           [bool(int(r["ok"])) for r in val]),
                               "calibrated": ece([r["calibrated_conf"] for r in val],
                                                 [bool(int(r["ok"])) for r in val])}
    save(model, MODEL)
    print(json.dumps({"thresholds": model["thresholds"], "val_ece": model["validation_ece"]}, indent=2))


def final_test(rows: list[dict], force: bool) -> None:
    if FINAL.exists() and not force:
        raise FileExistsError(f"{FINAL} exists; use --force only if you explicitly intend to invalidate the one-time test")
    model = load(MODEL)
    if data_hash([r for r in rows if r["split"] == "calib"]) != model["data_sha256"]:
        raise ValueError("Calibration data changed since fitting; refusing final test")
    test = [r for r in rows if r["split"] == "test"]
    if not test:
        raise ValueError("Test split is empty")
    for r in test:
        r["calibrated_conf"] = predict(model, r)
    report = {"n": len(test), "model_data_sha256": model["data_sha256"], "targets": {}}
    for target, thresholds in model["thresholds"].items():
        counts = {name: [r for r in test if route(r["pred"], r["calibrated_conf"], thresholds) == name]
                  for name in ("ACCEPT", "ESCALATE", "REVIEW")}
        accepted = counts["ACCEPT"]
        errors = sum(not int(r["ok"]) for r in accepted)
        report["targets"][target] = {"accepted": len(accepted), "coverage": len(accepted)/len(test),
                                     "observed_error": errors/len(accepted) if accepted else None,
                                     "error_95ci": wilson_interval(errors, len(accepted)) if accepted else None,
                                     "escalated": len(counts["ESCALATE"])/len(test),
                                     "review": len(counts["REVIEW"])/len(test)}
    FINAL.write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    correct = [bool(int(r["ok"])) for r in test]
    conf = [r["calibrated_conf"] for r in test]
    plt.figure()
    plt.plot([0, 1], [0, 1], "--", label="ideal")
    plt.scatter(conf, correct, alpha=0.35, label="test plates")
    plt.xlabel("Calibrated confidence")
    plt.ylabel("Correct (1) or wrong (0)")
    plt.legend()
    plt.savefig(config.RESULTS_DIR / "reliability.png")
    plt.close()
    points = risk_coverage(conf, correct)
    plt.figure()
    plt.plot(*zip(*points))
    plt.xlabel("Coverage")
    plt.ylabel("Observed error rate")
    plt.savefig(config.RESULTS_DIR / "risk_coverage.png")
    plt.close()
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = read_rows()
    if args.fit:
        fit_model(rows)
    else:
        final_test(rows, args.force)


if __name__ == "__main__":
    main()
