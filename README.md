# RiskGate ANPR

Indian plate OCR with calibrated confidence and selective routing. The current repository has 25 labeled Datacluster plates. The larger Kaggle set, calibrated model, protected final test, and Gemini subset results have **not** been produced yet.

## Current baseline

After row ordering, EasyOCR exact match on the 25 labeled plates is **5/25 (20%)**, up from 4/25. The chosen reader uses an A–Z/0–9 allowlist and no 2× upscale. CPU was used in the latest run because CUDA was unavailable.

## Commands

Run these from the project root with the existing `venv`:

```sh
make check
./venv/bin/python run_eval.py --no-upscale-variant
./venv/bin/python -m scripts.error_analysis
./venv/bin/streamlit run scripts/label_tool.py
```

After preparing `draft_labels.csv` and reviewing at least 200 readable plates in the label tool:

```sh
./venv/bin/python -m scripts.baseline_v2
./venv/bin/python -m scripts.eval_router --fit
./venv/bin/python -m scripts.eval_router --final
./venv/bin/python -m scripts.run_gemini_subset   # dry run: no real calls
./venv/bin/python -m scripts.run_gemini_subset --smoke --run
./venv/bin/python -m scripts.run_gemini_subset --run
./venv/bin/python -m scripts.eval_cascade
./venv/bin/streamlit run app/streamlit_app.py
```

Before `--run`, set `GEMINI_API_KEY` and a positive `GEMINI_MAX_CALLS` that fits your quota. The model is set in `riskgate/config.py`; check that it is available to your account. `run_gemini_subset.py` freezes a seeded manifest and reuses cached answers. The final test refuses to overwrite its output without `--force`; do not use `--force` for ordinary reruns.

The demo currently accepts a cropped plate. Detector training is deferred until the Kaggle annotations and GPU are available. See `DATA_NOTES.md` for data restrictions and remaining gates.
