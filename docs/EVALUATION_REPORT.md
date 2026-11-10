# RiskGate ANPR: measured evaluation

**Run date:** 10 November 2026  
**Hardware:** RTX 4060 Laptop GPU (8 GB); FastPlateOCR used ONNX CPU inference.  
**Project claim:** an offline Indian plate pipeline with measured accept/review behavior. These numbers describe an academic prototype, not deployment accuracy.

## Data and protocol

The [Saisirishan Indian vehicle dataset](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset) archive contains 1,697 image/XML pairs. Two annotations point to missing image files, leaving **1,695 usable labeled images**: 601 State-wise OLX, 440 Google images, and 654 video frames. XML object names provide plate text; XML boxes provide location. We extracted one crop per usable image.

The split keeps repeated plate text and frames from a source video in one group. It produced 847 detector/OCR-selection train images, 339 calibration images, 255 threshold-validation images, and 254 test images. The 25 labeled Datacluster plates are kept as a separate small source-shift check. The split manifest and generated data are local and Git-ignored.

I visually inspected a reproducible **180-crop sample** (60 from each source) using `results/audit_1.jpg` through `audit_9.jpg`. I logged **8 clear issues**: four foreign plates, one vehicle badge annotated as a plate, and three incomplete text labels. Three more crops were marked for closer inspection. These counts describe sampled images, not distinct vehicles; some video frames repeat a plate. The issues are listed in `results/audit_issues.csv`. This is a sampled inspection, **not a complete label correction**. All numerical results still use source XML text as ground truth. A larger correction pass is required before stronger claims.

## Components actually run

1. Fine-tuned a YOLO11n detector from free pretrained weights. The run was interrupted after three completed epochs; `best.pt` from those completed epochs was copied to `weights/plate_detector.pt` and used for the test. The training log is in `results/detector_training/`.
2. Compared existing EasyOCR and the free FastPlateOCR global model on identical labeled crops. FastPlateOCR was selected by training-split exact matches, before fitting confidence thresholds.
3. Fitted a logistic correctness model on 339 calibration predictions. Chose accept thresholds on 255 separate validation predictions. Evaluated the frozen thresholds on 254 grouped test images.
4. Ran a separate full-image detector → OCR → confidence → routing evaluation on those same test images. The original OCR test numbers had already been viewed, so the test was held out from model fitting but **not fully blinded**. No claim of a pristine one-time test is made.

## Results

| Component and split | n | Result |
| --- | ---: | --- |
| YOLO detector, test | 254 | 237 matched boxes at IoU ≥ 0.5; recall 93.3%; 246 boxes predicted, simplified precision 96.3% |
| EasyOCR, crop test | 254 | 36 exact plates (14.2%); mean CER 0.354 |
| FastPlateOCR, crop test | 254 | 71 exact plates (28.0%); mean CER 0.221 |
| FastPlateOCR, Datacluster cross-source sample | 25 | 4 exact plates (16.0%); too small for a general accuracy estimate |
| Full-image detector + FastPlateOCR, test | 254 | 71 exact plates (28.0%), 95% Wilson interval 22.8–33.8%; mean CER 0.258 |

The detector's localization is substantially better than the OCR's complete-text accuracy. A correct box is not a correct plate string. The full-image test counted missed detections as failed plate readings.

| Confidence and routing | Measured result |
| --- | --- |
| Validation ECE, raw OCR score | 0.442 |
| Validation ECE, calibrated estimate | 0.106 |
| Crop test ECE, raw OCR score | 0.559 |
| Crop test ECE, calibrated estimate | 0.075 |
| 2% and 5% target, crop test | 0/254 automatically accepted |
| 10% target, crop test | 72/254 accepted (28.3% coverage); 9/72 wrong (12.5% observed accepted error; 95% Wilson interval 6.7–22.1%) |
| 10% target, full-image test | 75/254 accepted (29.5% coverage); 11/75 wrong (14.7% observed accepted error) |

**Interpretation:** calibration improved average agreement between estimated confidence and observed correctness, but the chosen 10% target was **not met** on test. It is a target used to choose a threshold, not a guarantee. The strict targets selected no automatic answers, which is an honest result when evidence is too weak. Human review remains essential.

## Source slices and observed failure modes

| Test source | Images | Detector found a box | Exact full-image plate |
| --- | ---: | ---: | ---: |
| State-wise OLX | 90 | 85 | 14 |
| Google images | 50 | 48 | 16 |
| Video frames | 114 | 105 | 41 |

The video group and image-source mix affect the headline number. Some video groups contain many related frames, so percentages are not independent-vehicle accuracy. Low resolution, glare, two-row layouts, ambiguous characters, inaccurate XML labels, and non-Indian examples can all change OCR results. The local demo exposes plate crop and correction input so these failures are visible.

## Expanded detector and demo gallery (exploratory update)

The [Kedarsai box dataset](https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels) added 2,083 images locally. Only 181 were full vehicle JPGs suited to this detector; the 1,902 PNGs are mostly plate close-ups without verified text. One full image was excluded as a possible near-duplicate of the protected original holdout. The remaining **180** were added only to detector training, bringing its training count to **1,027**. Its bounding boxes were generated or checked by the dataset publisher, not by us. Twelve of its full photos were separately transcribed by visual inspection for the portable demo gallery; those examples are not used to claim independent OCR accuracy.

The detector was fine-tuned three more epochs from the original checkpoint. On the same 254-image grouped test, matched boxes rose **237 → 248** at IoU ≥ 0.5 (93.3% → 97.6% recall). Predicted boxes rose 246 → 268, so simplified precision fell 96.3% → 92.5%. The full pipeline still read **71/254** plates exactly; mean CER improved **0.258 → 0.237**. The accept/review counts remained 75 accepted with 11 errors. This is an **exploratory repeated test after earlier results were seen**, not a new blinded estimate. The updated app uses this expanded detector and exposes both versions in its evidence view.

Awiros was assessed as an optional reader. Its [model card](https://huggingface.co/Awiros/anpr-ocr) and [technical report](https://huggingface.co/Awiros/anpr-ocr/blob/main/TechnicalReport.pdf) describe a strong Indian specialist, but the model uses the Saisirishan dataset as seed training data and its provided inference script requires a separate PaddleOCR/PaddlePaddle runtime. Testing it on our current main dataset would not be independent, so it was not used for headline numbers or added to the offline demo.

## What to say in the faculty presentation

> “The detector found most plates, but exact character reading remained the hard part. I measured that limitation instead of hiding it. My confidence model made the system more selective: it can send uncertain results for human review. A validation target of 10% produced 14.7% accepted error in the complete held-out pipeline, so I would not deploy it unattended. The project demonstrates a measurable method and a working local system, and it identifies where more data and OCR training are needed.”

## Reproduce and inspect

- `results/baseline_v2_summary.txt`: both OCR readers by split.
- `results/final_test.json`: crop-based routing counts and interval.
- `results/detector_test.json`: detector localization counts.
- `results/end_to_end_test.json`: full-image accuracy and routing.
- `results/risk_coverage.png` and `results/reliability.png`: presentation plots.
- `results/audit_sample.csv`, `results/audit_issues.csv`, and `results/audit_*.jpg`: sampled label inspection.

These generated files and all source imagery remain local. The dataset's Kaggle metadata indicates CC BY-NC-ND 4.0; do not publish the source images, crops, or trained weights without checking the relevant terms.
