"""Offline comparison on the frozen Gemini subset; refuses missing cache entries."""

import csv
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from riskgate import config
from riskgate.cache import Cache, key
from riskgate.calibration import load, predict
from riskgate.engines.gemini_engine import PROMPT_VERSION
from riskgate.grammar import classify
from riskgate.metrics import normalize
from riskgate.router import route, wilson_interval


def main() -> None:
    manifest = json.loads((config.RESULTS_DIR / "gemini_subset.json").read_text())
    model = load(config.ROOT / "calibration.json")
    with (config.RESULTS_DIR / "baseline_v2.csv").open(newline="") as f:
        rows = {r["image"]: r for r in csv.DictReader(f)}
    cache = Cache(config.RESULTS_DIR / "gemini_cache")
    subset = []
    for image in manifest["images"]:
        crop = cv2.imread(image)
        if crop is None:
            raise FileNotFoundError(image)
        digest = key(cv2.imencode(".png", crop)[1].tobytes(), manifest["model"], PROMPT_VERSION)
        answer = cache.get(digest)
        if answer is None:
            raise ValueError(f"Missing Gemini cache for {image}; run the fixed subset first")
        if "latency_ms" not in answer:
            raise ValueError(f"Missing historical Gemini latency for {image}")
        subset.append((rows[image], answer))
    methods = {name: [] for name in ("Specialist only", "Gemini only", "Cascade")}
    agreements = []
    for row, gemini in subset:
        ok = bool(int(row["ok"]))
        ocr_ms = float(row["latency_ms"])
        gem_ms = float(gemini.get("latency_ms", 0))
        methods["Specialist only"].append((True, ok, 0, ocr_ms))
        gem_ok = normalize(gemini.get("text", "")) == normalize(row["truth"])
        gem_accepted = bool(gemini.get("text")) and not gemini.get("error")
        methods["Gemini only"].append((gem_accepted, gem_ok, 1, gem_ms))
        confidence = predict(model, row)
        initial = route(row["pred"], confidence, model["thresholds"]["0.05"])
        agree = gem_accepted and normalize(gemini["text"]) == normalize(row["pred"])
        accepted = initial == "ACCEPT" or (initial == "ESCALATE" and agree and classify(row["pred"]) != "INVALID")
        called = int(initial == "ESCALATE")
        methods["Cascade"].append((accepted, ok, called, ocr_ms + called * gem_ms))
        if row["split"] == "val" and initial == "ESCALATE" and agree:
            agreements.append(ok)
    if agreements:
        lo, hi = wilson_interval(len(agreements) - sum(agreements), len(agreements))
        agreement_note = (f"validation agreement group: {sum(agreements)}/{len(agreements)} correct "
                          f"(95% Wilson [{1-hi:.3f}, {1-lo:.3f}])")
    else:
        agreement_note = "no validation agreement cases"
    lines = ["# Frozen subset comparison", "", f"n={len(subset)}; {agreement_note}", "",
             "| Method | Accepted accuracy (95% Wilson) | Coverage | Gemini calls | Mean latency (ms) |",
             "| --- | --- | ---: | ---: | ---: |"]
    for name, values in methods.items():
        accepted = [v for v in values if v[0]]
        correct = sum(v[1] for v in accepted)
        n = len(accepted)
        lo, hi = wilson_interval(n - correct, n) if n else (0, 1)
        accuracy = f"{correct/n:.3f} [{1-hi:.3f}, {1-lo:.3f}]" if n else "n/a"
        lines.append(f"| {name} | {accuracy} | {n/len(values):.3f} | "
                     f"{sum(v[2] for v in values)} | {sum(v[3] for v in values)/len(values):.1f} |")
    (config.RESULTS_DIR / "cascade_table.md").write_text("\n".join(lines) + "\n")
    plt.figure()
    plt.bar(methods.keys(), [sum(v[0] for v in values) / len(values) for values in methods.values()])
    plt.ylabel("Coverage on frozen subset")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "cascade_coverage.png")
    plt.close()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
