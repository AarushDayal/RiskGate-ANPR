"""Selective routing with a Wilson upper error bound."""

import math

from riskgate.grammar import classify


TARGETS = (0.02, 0.05, 0.10)


def wilson_interval(errors: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = errors / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def choose_thresholds(rows: list[dict], review_below: float = 0.3) -> dict[str, dict]:
    eligible = sorted((r for r in rows if classify(r["pred"]) != "INVALID"),
                      key=lambda r: float(r["calibrated_conf"]), reverse=True)
    result = {}
    for target in TARGETS:
        chosen = None
        for i, row in enumerate(eligible):
            conf = float(row["calibrated_conf"])
            if i + 1 < len(eligible) and float(eligible[i + 1]["calibrated_conf"]) == conf:
                continue
            accepted = eligible[:i + 1]
            errors = sum(not int(r["ok"]) for r in accepted)
            if wilson_interval(errors, len(accepted))[1] <= target:
                chosen = conf
        result[f"{target:.2f}"] = {"accept_at": chosen, "review_below": review_below}
    return result


def route(text: str, confidence: float, thresholds: dict) -> str:
    if classify(text) != "INVALID" and thresholds["accept_at"] is not None and confidence >= thresholds["accept_at"]:
        return "ACCEPT"
    return "REVIEW" if confidence < thresholds["review_below"] else "ESCALATE"
