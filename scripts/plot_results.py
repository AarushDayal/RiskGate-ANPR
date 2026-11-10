"""Make presentation plots without changing the frozen evaluation JSON files."""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from riskgate import config
from riskgate.calibration import load, predict
from riskgate.metrics import risk_coverage


def rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def style() -> None:
    plt.rcParams.update({"figure.facecolor": "#0c1c2c", "axes.facecolor": "#10283b",
                         "axes.edgecolor": "#42616f", "axes.labelcolor": "#d8edf0",
                         "text.color": "#e8f7f5", "xtick.color": "#a9cbd1", "ytick.color": "#a9cbd1",
                         "grid.color": "#27485a", "font.size": 11})


def main() -> None:
    style()
    model = load(config.ROOT / "calibration.json")
    cropped = [row for row in rows(config.RESULTS_DIR / "baseline_v2.csv") if row["split"] == "test"]
    full = rows(config.RESULTS_DIR / "end_to_end_details.csv")
    crop_conf = [predict(model, row) for row in cropped]
    crop_ok = [bool(int(row["ok"])) for row in cropped]
    full_conf = [float(row["confidence"]) for row in full]
    full_ok = [bool(int(row["ok"])) for row in full]

    fig, ax = plt.subplots(figsize=(7, 4.3))
    for confidence, correct, title, color in ((crop_conf, crop_ok, "Labeled crop", "#5fe6ca"),
                                               (full_conf, full_ok, "Full vehicle image", "#ffba6b")):
        points = risk_coverage(confidence, correct)
        ax.plot([x for x, _ in points], [y for _, y in points], color=color, linewidth=2.5, label=title)
    ax.set(xlabel="Fraction handled automatically (coverage)", ylabel="Wrong among accepted (risk)",
           title="More automation can mean more mistakes", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.4)
    ax.legend(facecolor="#173449", edgecolor="#42616f", labelcolor="#e8f7f5")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "risk_coverage.png", dpi=160)
    plt.close(fig)

    bins = [i / 8 for i in range(9)]
    means, accuracy, count = [], [], []
    for left, right in zip(bins[:-1], bins[1:]):
        indexes = [i for i, value in enumerate(crop_conf) if left <= value < right or right == 1 and value == 1]
        if indexes:
            means.append(sum(crop_conf[i] for i in indexes) / len(indexes))
            accuracy.append(sum(crop_ok[i] for i in indexes) / len(indexes))
            count.append(len(indexes))
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot([0, 1], [0, 1], "--", color="#9db7c0", label="Perfect calibration")
    ax.scatter(means, accuracy, s=[max(40, n*4) for n in count], color="#5fe6ca", alpha=.85,
               edgecolor="#d8fff5", label="Test bins; size = sample count")
    ax.set(xlabel="Estimated chance of exact plate", ylabel="Observed exact-match frequency",
           title="Does stated confidence match reality?", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.4)
    ax.legend(facecolor="#173449", edgecolor="#42616f", labelcolor="#e8f7f5")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "reliability.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
