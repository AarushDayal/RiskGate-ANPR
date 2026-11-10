"""Freeze a seeded validation/test subset and optionally spend Gemini quota once."""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2

from riskgate import config
from riskgate.engines.gemini_engine import GeminiEngine, PROMPT_VERSION


MANIFEST = config.RESULTS_DIR / "gemini_subset.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="make real calls up to GEMINI_MAX_CALLS")
    parser.add_argument("--smoke", action="store_true", help="process first 10 validation crops")
    parser.add_argument("--val", type=int, default=60)
    parser.add_argument("--test", type=int, default=90)
    args = parser.parse_args()
    with (config.RESULTS_DIR / "baseline_v2.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == "kaggle"]
    if not any(r["split"] == "val" for r in rows) or not any(r["split"] == "test" for r in rows):
        raise ValueError("Reviewed Kaggle validation and test plates are required")
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        if manifest["model"] != config.GEMINI_MODEL or manifest["prompt_version"] != PROMPT_VERSION:
            raise ValueError("Existing subset uses a different model or prompt; do not spend quota silently")
    else:
        rng = random.Random(config.SEED)
        selected = []
        for split, count in (("val", args.val), ("test", args.test)):
            pool = [r["image"] for r in rows if r["split"] == split]
            rng.shuffle(pool)
            selected.extend(pool[:count])
        manifest = {"seed": config.SEED, "model": config.GEMINI_MODEL,
                    "prompt_version": PROMPT_VERSION, "images": selected}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    images = manifest["images"]
    if args.smoke:
        val_images = {r["image"] for r in rows if r["split"] == "val"}
        images = [p for p in images if p in val_images][:10]
    engine = GeminiEngine(run=args.run)
    failures = 0
    for image in images:
        crop = cv2.imread(image)
        if crop is None:
            raise FileNotFoundError(image)
        result = engine.read(crop)
        failures += bool(result.error and result.error != "dry run")
    print(json.dumps({"subset": len(images), "real_calls": engine.calls_used,
                      "would_call": engine.would_call, "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
