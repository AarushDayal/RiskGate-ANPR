# RiskGate ANPR: current state

**Checked:** 10 November 2026  
**Project root:** `/home/aarush/Aarush/Projects/riskgate`  
**Full results and caveats:** [EVALUATION_REPORT.md](EVALUATION_REPORT.md)

## Working now

- 1,695 usable Indian-vehicle-dataset image/XML pairs imported locally, with crops and a grouped train/calibration/validation/test split. Two annotations had missing images.
- EasyOCR and FastPlateOCR compared on the same crops. FastPlateOCR selected on training data and used for the final pipeline.
- A YOLO11n plate detector fine-tuned on the laptop's RTX 4060. An exploratory second stage added 180 screened full vehicle photos and matched 248/254 test boxes at IoU ≥ 0.5, versus 237/254 before expansion.
- A logistic correctness estimator fitted on calibration data and accept/review thresholds chosen on validation data. The 10% target accepted 75/254 full images with 11 wrong; 2% and 5% targets accepted none.
- An offline Streamlit console with full-image and crop modes, 12 manually transcribed local gallery photos, batch review and CSV export, policy selection, decision trace, correction form, evidence charts, and methodology tab.
- 23 automated tests pass. The demo renders under Streamlit's app test utility.
- A 180-crop visual audit sample was inspected. It found eight clear issues and three more suspect crops; the XML labels remain unverified as a whole.

## Current limitation

The complete image-to-text accuracy is 71/254 (28.0%) on the grouped test. OCR is much weaker than plate localization. The 10% target was exceeded on test, so the system must keep human review. The test was held out from fitting but its crop OCR statistics were viewed before the routing evaluation, and the expanded detector was evaluated on that same test. See the report for interval estimates and source slices.

## Files and commands

- `scripts/prepare_indian.py`: parse XML, create crops, group repeated plates and videos.
- `scripts/import_extra_detection.py`: screen and add 180 box-only full photos without mixing in protected holdouts.
- `scripts/prepare_gallery.py`: copy and label 12 portable demo examples.
- `scripts/baseline_v2.py`: local OCR comparison and reader selection.
- `scripts/eval_router.py`: calibrate, set thresholds, guarded crop test.
- `scripts/detector.py`: detector preparation, training, and test localization.
- `scripts/eval_end_to_end.py`: guarded full-image test.
- `scripts/plot_results.py`: presentation plots.
- `app/streamlit_app.py`: local demo. Start with `./venv/bin/streamlit run app/streamlit_app.py`.

Generated data, weights, and results are excluded from Git. The root `README.md` has rebuild instructions. Earlier plans and walkthroughs are in this `docs/` directory.
