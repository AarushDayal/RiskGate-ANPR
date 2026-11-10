"""Small logistic correctness model, stored as portable JSON."""

import hashlib
import json
import math
from datetime import date
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from riskgate.features import FEATURE_NAMES


def data_hash(rows: list[dict]) -> str:
    content = json.dumps(sorted(rows, key=lambda r: (r.get("source", ""), r.get("image", ""))),
                         sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def fit(rows: list[dict]) -> dict:
    if len(rows) < 20 or len({int(r["ok"]) for r in rows}) != 2:
        raise ValueError("Calibration needs at least 20 reviewed reads with both correct and wrong examples")
    x = [[float(r[name]) for name in FEATURE_NAMES] for r in rows]
    y = [int(r["ok"]) for r in rows]
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=0.1, max_iter=1000).fit(scaler.transform(x), y)
    return {"feature_names": FEATURE_NAMES, "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]), "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "data_sha256": data_hash(rows),
            "date": date.today().isoformat()}


def predict(model: dict, features: dict[str, float]) -> float:
    score = model["intercept"] + sum(
        w * (float(features[name]) - mean) / scale
        for name, w, mean, scale in zip(model["feature_names"], model["coefficients"],
                                        model["mean"], model["scale"]))
    return 1 / (1 + math.exp(-max(-700, min(700, score))))


def save(model: dict, path: Path) -> None:
    path.write_text(json.dumps(model, indent=2) + "\n")


def load(path: Path) -> dict:
    return json.loads(path.read_text())
