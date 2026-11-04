import re


def normalize(text: str) -> str:
    """Uppercase, keep only A-Z and 0-9."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def exact_match(pred: str, truth: str) -> bool:
    return normalize(pred) == normalize(truth)


def cer(pred: str, truth: str) -> float:
    """CER = edit distance / len(truth). Always in [0, inf); sane equals converters -> tiny CER.

    # ponytail: O(n*m) DP is fine at ~10 chars per plate
    """
    p, t = normalize(pred), normalize(truth)
    if not t:
        return 0.0 if not p else 1.0
    d = list(range(len(t) + 1))
    for i, cp in enumerate(p, 1):
        prev, d[0] = d[0], i
        for j, ct in enumerate(t, 1):
            next_v = min(d[j] + 1, d[j - 1] + 1, prev + (cp != ct))
            prev, d[j] = d[j], next_v
    return d[-1] / len(t)


def ece(confs: list[float], corrects: list[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error: sum over bins of weight * |accuracy - mean confidence|."""
    assert len(confs) == len(corrects)
    e = 0.0
    for b in range(n_bins):
        idx = [i for i, c in enumerate(confs) if min(int(c * n_bins), n_bins - 1) == b]
        if not idx:
            continue
        acc = sum(corrects[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        e += len(idx) / len(confs) * abs(acc - conf)
    return e


def risk_coverage(confs: list[float], corrects: list[bool]) -> list[tuple[float, float]]:
    """Sort by confidence desc; after each new plate return (coverage, risk) = (k/N, error rate in top-k)."""
    order = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    points, wrong = [], 0
    for k, i in enumerate(order, 1):
        wrong += not corrects[i]
        points.append((k / len(confs), wrong / k))
    return points
