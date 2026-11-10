# RiskGate ANPR: simple project and viva guide

**Deadline:** 16 November 2026  
**Use this honestly:** the current full-image test read **71/254** plates exactly. The expanded detector matched **248/254** labeled boxes (the original matched 237/254); OCR is the limiting stage. At the 10% validation target, the complete system accepted **75/254**, and **11 of those 75 were wrong**. The detector update reused a previously viewed test. See [the evaluation report](EVALUATION_REPORT.md) for conditions and limitations. The old EasyOCR baseline was **5/25** on a small Datacluster sample.

## The project in one sentence

RiskGate reads an Indian vehicle number plate from a photo and decides whether to **accept the reading automatically or ask a person to check it**.

ANPR means *Automatic Number Plate Recognition*. The useful feature is the decision about when to trust the reading. A wrong plate can look plausible, so showing text alone is not enough.

## A plain example

Suppose a photo contains a car with the true plate `KA03MN1234`.

1. **Detection** finds the plate's rectangle inside the full photo and crops it.
2. **OCR** (optical character recognition) reads the letters and digits. It might return `KA03MN1234`, or make a subtle mistake such as `KA03MN1284`.
3. **Grammar check** asks whether the result resembles an Indian plate. It cannot prove the text is right: both examples look plausible.
4. **Confidence model** combines OCR score, grammar result, crop quality, and other simple signals to estimate the chance that the *whole plate* is right.
5. **Router** accepts a strong prediction or sends an uncertain one to a review screen. A person can correct the text. A Gemini second opinion is optional and must not be required for the demo.

The screen should show the original image, detected box, crop, proposed text, confidence, decision, and reason. This lets an examiner see every step.

## Why the first Gemini prototype was not enough

The earlier prototype sent a photo to Gemini and displayed a response. It showed that recognition was possible, but did not measure how often the answer was wrong, whether the system knew when it was wrong, or whether it could work without an API key. The final project adds a local pipeline, an Indian dataset, a trained detector if time permits, a separate final test, and an accept/review decision. Its main contribution is **measured selective prediction**, not inventing a new OCR network.

## What each part really does

| Part | Easy meaning | What we build or use | One limitation to remember |
| --- | --- | --- | --- |
| Data | Photos plus correct plate text and location | 1,697 image/XML pairs from the [Indian vehicle dataset](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset); inspect labels before using them | An XML label can be wrong, and video frames may repeat the same vehicle |
| Detector | Finds where the plate is | Fine-tune a small YOLO model on labeled rectangles, if feasible | It may miss a plate or crop it badly |
| OCR | Reads the crop | Compare existing EasyOCR with one local specialist | `O` versus `0`, blur, glare, and two-row layouts cause mistakes |
| Grammar | Checks whether text has a plausible form | Existing Indian plate classifier | A plausible-looking plate can still be wrong |
| Calibration | Turns model signals into an estimated chance of a correct whole plate | Existing logistic regression code, fitted on real labeled examples | It must be tested on data it did not fit |
| Router | Decides accept or human review | Choose threshold on validation data | Higher caution means fewer automatic accepts |
| Review | Lets a person resolve uncertain cases | Correction field in demo | Human review takes time |

**Training versus using:** fine-tuning a detector means starting with an existing model and teaching it where Indian plates appear. It does not mean training AI from zero. Running EasyOCR uses a pre-trained reader. Fitting the small confidence model teaches it the relation between signals and correct/incorrect reads. State exactly which of these actually happened in the final report.

## The three decisions you should be able to explain

### 1. Why local models?

They work without network access or a paid key, make the demo repeatable, and let us measure speed. A free Gemini quota can change. An optional Gemini comparison is useful research, but the main claim should survive with zero calls.

### 2. Why send some results to review?

In real use, an incorrect automatic plate record may be more costly than asking a person to check a difficult image. The system trades **coverage** (fraction accepted automatically) against **risk** (fraction wrong among accepted readings). For example, if it accepts 60 of 100 plates and 3 accepted readings are wrong, coverage is 60% and observed accepted error is 3/60 = 5%. The 40 review cases still need a person; they are not counted as automatic successes.

### 3. Why measure confidence separately?

An OCR score of 0.90 is often just a model score. It does not automatically mean “90% chance the whole plate is right.” A calibrated score of 0.90 should mean that, across many comparable cases given about 0.90, roughly 90% are truly right. We learn this relationship on labeled data and check it on separate data. A nice confidence plot is evidence, not proof of a guarantee on every image.

## How the evaluation works

1. **Check labels.** We visually inspected 180 of 1,695 source annotations, spread evenly across the three sources. Eight clear issues were logged, including foreign plates and incomplete labels. The original XML text still supplies the numerical ground truth, so the data is not fully corrected. OCR output is never the answer key.
2. **Split by vehicle/video group.** Similar frames of one vehicle, and repeated copies of the same plate, stay in one group. Otherwise the test would be too easy because the model had effectively seen that car before. Save a split manifest so results can be reproduced.
3. **Train and choose using development data.** Train the detector only on its training group. Compare OCR readers on development/validation data. Fit confidence on calibration data. Pick the review threshold on separate validation data.
4. **Open the final test once.** It measures the frozen system. A missed detector box counts as an end-to-end failure, even if crop-only OCR works well.

Useful measures:

| Measure | What to say in the viva |
| --- | --- |
| Exact plate match | “The entire predicted plate must match the labeled plate after consistent normalization.” |
| Character error rate (CER) | “Number of character insertions, deletions, and replacements needed, divided by true plate length. It shows how far a wrong reading is from the answer.” |
| Detection precision/recall | “Precision asks how many found boxes are real plates; recall asks how many real plates were found.” |
| mAP | “A standard detector score that combines correctness across box overlap thresholds. I will show precision and recall too because they are easier to interpret.” |
| Calibration/ECE | “Checks whether stated confidence roughly matches how often predictions are correct.” |
| Coverage and accepted error | “How much work we automate, and how many of those automatic answers are wrong.” |
| Latency | “Time taken per image; measure locally and report the hardware.” |

Always put the **number of images** beside a percentage. A 5% target is not a proven 5% guarantee: the test set is finite, and selecting thresholds on data introduces uncertainty. Report a confidence interval and say what was actually observed.

## Research papers: one line each for the presentation

| Source | What we learned | How it affects RiskGate |
| --- | --- | --- |
| [BLPR](https://arxiv.org/html/2604.09927v2) | Use an expensive VLM only on uncertain cases | Optional second opinion rather than calling Gemini for every photo |
| [Neural Sentinel](https://arxiv.org/pdf/2602.07051) | Calibrate confidence and collect human corrections | Confidence plot and review screen |
| [Nigerian VLM comparison](https://arxiv.org/pdf/2607.02025) | Compare readers on the same images using exact match and CER | Fair local OCR comparison |
| [VehiclePaliGemma](https://arxiv.org/html/2412.14197v1) | Synthetic data can help training, but real labeled images must judge performance | Real Indian test is the headline |
| [ICPR low-resolution competition](https://arxiv.org/html/2604.22506v2) | Tiny and blurry plates are hard; video can provide multiple views | Show low-resolution failures; multi-frame fusion is future work |
| [Indian Licence Plate Dataset in the Wild](https://arxiv.org/pdf/2111.06054) | Indian plates vary in layout and image conditions | Use Indian data and inspect difficult crops |
| [Awiros report](https://huggingface.co/Awiros/anpr-ocr/blob/main/TechnicalReport.pdf) | Domain-specific OCR can be strong, but benchmark provenance matters | Its training used our proposed Kaggle dataset, so we cannot claim an independent Awiros test on those images |

Do not repeat another paper's accuracy as *our* accuracy. Explain which idea influenced our design.

## A simple two-minute presentation script

> “My first prototype asked Gemini to read a plate from a photo. It worked as a demonstration, but I could not tell when its answer was wrong, and it depended on an external API. For the final project, I changed the question: can a local Indian ANPR system decide when to accept its own reading and when to ask for review?
>
> The pipeline detects the plate in a full image, crops it, reads the crop with OCR, checks the plate format, estimates the chance that the entire text is correct, and routes uncertain cases to a person. I used a public Indian dataset with plate locations and text. I kept similar frames and repeated plate identities together when splitting data, so the final test is more honest.
>
> Here are the measured detector and OCR results [point to actual tables]. This graph shows the tradeoff: as I accept more plates automatically, mistakes among accepted plates can rise. Here are examples where the system succeeded and failed. The demo works locally, while Gemini is only an optional comparison. My main result is the measured accept-versus-review behavior, not a claim of perfect reading.”

Replace the bracketed sentence with the actual result: “The detector matched 237 of 254 boxes, while the complete system read 71 of 254 whole plates exactly.” Explain that OCR, not localization, is the limiting stage.

## Likely faculty questions and clear answers

**What is new here if OCR already exists?**  
“I combined detection, local OCR, confidence calibration, and review routing for Indian plates, then measured the full system on a separate test. The contribution is knowing when to use its answer, rather than claiming a new OCR algorithm.”

**Why was your first EasyOCR result only 5/25?**  
“Those 25 examples form a tiny, difficult baseline, with some crop and layout problems. I include it honestly as motivation. The larger, separated evaluation is what will support any final claim.”

**Could the format check prove the plate is correct?**  
“No. `KA03MN1234` and `KA03MN1284` both look valid. Grammar is one warning signal, not a substitute for the image or ground truth.”

**Why not send every case to Gemini?**  
“That reintroduces key, quota, latency, and network dependence. Selective review makes the local result useful and makes API cost measurable if a second opinion is added.”

**How did you avoid data leakage?**  
“We grouped video frames and repeated plate identities before splitting, then froze the model and threshold before the final test. I can show the manifest and group checks.” Only say this after those files have been created and checked.

**Does 90% confidence mean nine in ten of these are correct?**  
“Only if calibration on held-out data supports it. Raw OCR confidence does not have that interpretation.”

**What happens if the detector misses?**  
“The full-image prediction fails and is counted as an end-to-end error. The demo can also accept a crop to show the later stages separately.”

**What are the limits?**  
“The dataset is modest, may contain label errors and near-duplicates, and may not cover every Indian plate style or night/weather condition. Human review is still needed for uncertain cases. Results apply to the tested data, not all traffic cameras.”

**What did you personally do?**  
Describe only your actual role: choosing the problem, inspecting data, reviewing labels, running experiments, checking failures, building or integrating the pipeline, and interpreting results as applicable. If you used AI coding help, say so plainly. The faculty can test understanding by changing an example; being able to explain each step is more valuable than claiming to have written every line without help.

## Fast study method

Spend **25–30 minutes a day** on this guide and the actual output files. First, trace one image through all steps without notes. Next, explain the difference between OCR score and calibrated correctness. Then explain coverage versus accepted error using the 60/100 example. Finally, practice the script with two real successes and two real failures. If you can explain those four things, you can discuss the project without memorizing model jargon.

Keep a tiny experiment log: date, data version, command, setting changed, result, one failure learned. It gives you truthful evidence of progress and makes the viva easier.
