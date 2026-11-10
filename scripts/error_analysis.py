"""Categorize incorrect plate reads from the baseline CSV."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from riskgate.metrics import normalize


CONFUSABLE = {"0": "O", "O": "O", "1": "I", "I": "I", "8": "B", "B": "B",
              "5": "S", "S": "S", "2": "Z", "Z": "Z"}


def category(pred: str, truth: str) -> str:
    pred, truth = normalize(pred), normalize(truth)
    if not pred:
        return "empty"
    if len(pred) == len(truth) and sorted(pred) == sorted(truth):
        return "order_swapped"
    if len(pred) == len(truth) and all(CONFUSABLE.get(a, a) == CONFUSABLE.get(b, b)
                                         for a, b in zip(pred, truth)):
        return "confusable"
    if truth and truth in pred:
        return "extra_text"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/baseline_easyocr.csv"))
    args = parser.parse_args()
    with args.input.open(newline="") as f:
        wrong = [r for r in csv.DictReader(f) if normalize(r["pred"]) != normalize(r["truth"])]
    for row in wrong:
        row["category"] = category(row["pred"], row["truth"])
    out = Path("results")
    out.mkdir(exist_ok=True)
    with (out / "errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*wrong[0].keys()] if wrong else ["category"])
        writer.writeheader()
        writer.writerows(wrong)
    counts = Counter(r["category"] for r in wrong)
    summary = "\n".join(f"{name}: {counts[name]}" for name in
                        ("empty", "order_swapped", "confusable", "extra_text", "other"))
    (out / "error_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
