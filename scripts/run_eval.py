"""Refactored eval (from eval_easyocr.py): N variants on all labelled plates -> vs table + CSV."""

import argparse
import csv
from pathlib import Path

import cv2

from riskgate import config
from riskgate.data import crop, load_samples
from riskgate.engines.easyocr_engine import EasyOCREngine
from riskgate.grammar import classify
from riskgate.metrics import cer, exact_match, normalize


def run_variant(engine, samples, images_limit=None) -> list[dict]:
    if images_limit:
        keep = list(dict.fromkeys(Path(s.image_path).name for s in samples))[:images_limit]
        samples = [s for s in samples if Path(s.image_path).name in keep]
    rows = []
    for s in samples:
        img = cv2.imread(s.image_path)
        c = crop(img, s.bbox) if img is not None else None
        if c is None:
            continue
        r = engine.read(c)
        pred = "" if r.error else normalize(r.text)
        truth = normalize(s.truth)
        rows.append(
            dict(
                image=Path(s.image_path).name,
                truth=truth,
                pred=pred,
                conf=round(r.raw_conf, 3),
                grammar_class=classify(pred),
                ok=int(exact_match(pred, truth)),
                cer=round(cer(pred, truth), 3),
            )
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    good = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    avg = lambda L, k: round(sum(r[k] for r in L) / len(L), 3) if L else None
    return dict(
        n=n,
        exact_match=round(len(good) / n, 3) if n else 0,
        mean_cer=avg(rows, "cer"),
        conf_correct=avg(good, "conf"),
        conf_wrong=avg(bad, "conf"),
        valid_correct=avg(good, "grammar_valid"),
        valid_wrong=avg(bad, "grammar_valid"),
    )


def table(summaries: dict[str, dict]):
    cols = ["n", "exact_match", "mean_cer", "conf_correct", "conf_wrong", "valid_correct",
            "valid_wrong"]
    w = max(len(t) for t in summaries) + 2
    print(f'{"variant":{w}}' + "".join(f"{c:>14}" for c in cols))
    for tag, s in summaries.items():
        print(f"{tag:{w}}" + "".join(f"{str(s[c]):>14}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, help="limit to first N images (smoke run)")
    ap.add_argument("--no-upscale-variant", action="store_true")
    args = ap.parse_args()
    config.device()
    samples = load_samples()
    variants = {
        "baseline": dict(),
        "allowlist": dict(allowlist=True),
    }
    if not args.no_upscale_variant:
        variants["allowlist+upscale"] = dict(allowlist=True, upscale=True)

    summaries, all_rows = {}, {}
    for tag, cfg in variants.items():
        engine = EasyOCREngine(**cfg)
        rows = run_variant(engine, samples, args.images)
        all_rows[tag] = rows
        for r in rows:
            r["grammar_valid"] = int(r["grammar_class"] == "VALID")
        summaries[tag] = summarize(rows)
        table({tag: summaries[tag]})

    print("\n=== comparison (exact_match, mean CER, mean conf correct/wrong, grammar-valid correct/wrong) ===")
    table(summaries)

    # smoke runs must not clobber the full-run CSV
    if args.images is None:
        baseline = all_rows["baseline"]
        config.RESULTS_DIR.mkdir(exist_ok=True)
        out = config.RESULTS_DIR / "baseline_easyocr.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["image", "plate_index", "truth", "pred", "conf", "grammar_class", "ok"],
                extrasaction="ignore",
            )
            w.writeheader()
            w.writerows([dict(**r, plate_index=i) for i, r in enumerate(baseline)])
        print(f"\nbaseline per-plate CSV -> {out}")


if __name__ == "__main__":
    main()
