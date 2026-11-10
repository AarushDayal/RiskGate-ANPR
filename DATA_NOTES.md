# Data notes

- **Datacluster sample:** 47 images, 25 labeled plate instances. Its local `datacluster/README.md` declares CC BY-NC-ND 4.0 and evaluation-only use. Do not publish these images or crops.
- **Kaggle source:** [kedarsai/indian-license-plates-with-labels](https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels). It is not downloaded here. The exact license and annotation layout still need checking against the downloaded dataset before preparing crops or sharing artifacts. Do not assume its images can be redistributed.
- **Review status:** no Kaggle text labels yet. Target: 150–250 readable Kaggle plates plus the existing 25. Recheck a random 10% of reviewed labels and record the checked count and error count here.
- **Leakage control:** `splits_v2.json` groups plates by source image. Kaggle supplies calibration, validation, and test groups; Datacluster is held out as a cross-dataset check. The split is unavailable until reviewed Kaggle labels exist.
- **Protected results:** `calibration.json`, `results/final_test.json`, and Gemini subset outputs are not present. No Phase 2 or Phase 3 performance claim is available yet.
