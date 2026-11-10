# RiskGate explained from the beginning

**A plain-language guide to what we built, how it works, and why.**  
**Project state checked:** 11 November 2026. For the precise experiment conditions, see the [evaluation report](EVALUATION_REPORT.md).

## The whole project in one minute

RiskGate takes a photo of a vehicle and tries to read its number plate. It then decides whether to **accept the reading automatically** or **ask a person to check it**. ANPR stands for *Automatic Number Plate Recognition*. OCR stands for *optical character recognition*: reading letters and numbers from an image.

The key idea is simple: a computer can give a convincing-looking answer that is wrong by just one character. RiskGate therefore shows the evidence behind its answer and sends uncertain cases for review.

```text
Vehicle photo
    ↓
Find the plate's location (detector)
    ↓
Cut out the plate (crop)
    ↓
Read its letters and digits (OCR)
    ↓
Check its format and estimate whether the whole reading is correct
    ↓
ACCEPT, or ask a person to REVIEW and correct it
```

There are **two separate processes** to understand. *Building and evaluating* the system happens beforehand, using labeled images. *Using* the system happens when someone uploads a new photo to the demo. Training does not happen every time a photo is uploaded.

## Part 1: How we built the system

### Step 1 — Get images with answers attached

The main [vehicle dataset](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset) contains photos and XML annotation files. Think of an XML file as a small answer sheet for a photo. It gives the plate's rectangle and the text that the plate is supposed to contain. We found **1,695 usable image/XML pairs**; two more annotations pointed to missing photos.

The labels are helpful, but they are not perfect. We visually inspected 180 plate crops and found eight clear annotation problems, including foreign plates, incomplete text, and one vehicle badge labeled as a plate. The numerical results still use the source XML text as their answer key. That limits how strongly we can claim accuracy. The [data notes](DATA_NOTES.md) explain the sources and label status.

### Step 2 — Keep similar photos together when splitting the data

We divided the 1,695 images into four groups:

| Group | Images | What it is for |
| --- | ---: | --- |
| Training | 847 | Teach the detector and select the local OCR reader. |
| Calibration | 339 | Learn what correct and incorrect OCR results tend to look like. |
| Validation | 255 | Choose the rule for automatic acceptance. |
| Test | 254 | Measure the resulting system on images not used for fitting. |

Many photos came from video. Ten adjacent frames might show the same car and plate. If some frames were used for training and others for testing, the test would be too easy. We therefore grouped frames from the same source video and repeated plate identities before splitting.

This reduces a common problem called **data leakage**: accidentally letting the system see nearly the same example while it learns and while it is being tested. It does not rule out every possible duplicate. Also, some test results were viewed before later detector improvements, so the final comparison is *held out from fitting but not fully blinded*.

### Step 3 — Train a model to find the plate

The detector's only job is to answer: **“Where in this full photo is the number plate?”** It draws a rectangle around the most likely plate. We started with a small, already-pretrained YOLO11n model and fine-tuned it on plate rectangles. This is much more practical on a laptop RTX 4060 than training a vision model from nothing.

We also screened a second dataset and added **180 full vehicle photos with plate boxes** to detector training. The detector then had **1,027 training images** in total. Those extra images did not have trustworthy plate text, so we did not treat them as OCR answers.

On the 254-image test, the expanded detector produced a matching box for **248 images**, using a minimum box overlap of 0.5. It found some sort of plate box in 250 images, but two of those boxes did not match the labeled location closely enough. A *correct box* tells us where the plate is; it does **not** tell us that every character will be read correctly.

### Step 4 — Choose a model to read the cropped plate

Once the plate is located, the app cuts out just that part of the photo. This smaller image is the **crop**. We compared two local OCR readers, EasyOCR and FastPlateOCR, on the same labeled plate crops. FastPlateOCR was chosen because it read more complete plates correctly in the development comparison. The final demo uses its local model; it does not call Gemini.

We did **not** train a new OCR network from scratch. We integrated an existing reader and measured its behavior on this data. That distinction matters when explaining what work was actually done.

### Step 5 — Learn whether a whole reading is likely to be right

An OCR model supplies a score, but a high OCR score does not automatically mean “this entire plate is correct.” For example, `KA03MN1234` and `KA03MN1284` differ by one digit, yet both look believable.

We used the 339 calibration images to train a small **logistic regression** model. For each OCR result, it looks at the OCR score, whether the text resembles a plate format, text length, crop width and height, and OCR detection information. It learns how those signals relate to a **completely correct plate reading**. The resulting number is called the *correctness estimate*.

This is called **calibration**. If many similar readings receive an estimate near 80%, roughly 80% of those readings should be correct for the estimate to be well calibrated. It is still an estimate, not proof that any particular plate is right. The format check is only one clue: a wrong plate can have a perfectly plausible format.

### Step 6 — Choose when to accept and when to review

We used the separate 255-image validation group to choose confidence thresholds for three intended error targets: 2%, 5%, and 10%. The rule accepts a reading only when its format is eligible and its correctness estimate reaches the chosen threshold. Otherwise, the app marks it **REVIEW**. In the offline demo, a possible “get another opinion” state also ends in human review; it does not make an API call.

The tradeoff is:

- **Coverage:** How many images the computer handles automatically? If it accepts 30 of 100 images, coverage is 30%.
- **Accepted error:** Of the accepted readings, how many are wrong? If 3 of those 30 are wrong, accepted error is 3/30 = 10%.

Increasing caution generally reduces the number accepted. A “10% target” describes how a threshold was **selected on validation data**. It is not a guarantee of 10% error on new photos.

### Step 7 — Measure the complete system

Finally, we ran full vehicle photos through **detection → crop → OCR → confidence → decision**. If the detector misses the plate, that photo counts as an end-to-end failure, even if the OCR reader might have succeeded on a manually supplied crop. This is the result that matters when someone uploads a whole vehicle photo.

The 12 photos in the demo gallery are full vehicle or street scenes for showing how the app works. They are **presentation examples**, not a new benchmark or extra evidence of accuracy.

## Part 2: What happens to one photo in the demo

Imagine you upload a photo of a car whose true plate reads `KA03MN1234`. These characters are just an example; they are not a claimed test result.

1. **The app receives the photo.** You can upload a JPG/PNG or select a gallery scene. The main demo processes it on this machine.
2. **The detector proposes a rectangle.** The app shows that box on the original image. If it finds no plate, it reports that the image needs manual review.
3. **The app cuts out the rectangle.** You see the crop beside the result. A poor crop can already make the next step fail.
4. **OCR reads the crop.** It might return `KA03MN1234`, or make a subtle error such as `KA03MN1284`.
5. **The app checks and scores the result.** It checks the plate's format and estimates the chance that the *entire string* is correct. Both example strings could pass a simple format check, so format alone cannot settle the case.
6. **The app routes the result.** A result that meets the selected acceptance rule is shown as **ACCEPT**. Other results are shown as **REVIEW** so a person can inspect the photo and correct the text.

The **Plate crop** input mode skips detection and sends a supplied crop straight to OCR. This helps demonstrate whether a failure comes from *finding* the plate or *reading* it. It is a diagnostic mode, not the complete vehicle-photo pipeline.

The app also offers a batch view for up to 20 uploads and a CSV export of its decisions. A human correction is saved in a local log. **Saving a correction does not automatically retrain the models**; it records information that could be reviewed and used in a future training cycle.

## Why this architecture instead of only calling an LLM API?

Your original prototype sent a photo to Gemini and displayed its answer. That was useful as a first demonstration: it proved that an AI system could sometimes read a plate. It did not establish how often the answer was exactly right, whether a plausible answer could be trusted, or how the demo would work without an API key or internet connection.

RiskGate separates the task into visible jobs. We can measure **plate location**, **character reading**, and **accept/review behavior** independently. The complete pipeline runs locally, so the main demo does not depend on paid API access or sending a vehicle photo to an external service.

An image-capable LLM could be added as an **optional second opinion** for difficult cases. That would produce another possible reading; it would not automatically tell us which reading is correct. If two readers disagree, a person still needs to check. An API could also add network delay, quota or cost limits, and data-sharing considerations. RiskGate currently uses **human review**, not an LLM API, for uncertain cases.

We also examined the Awiros Indian OCR model. Its published material describes use of our main dataset as seed training data, so testing it on that same dataset would not be an independent comparison. Its inference setup also needs another runtime. We therefore did not add it to the headline result or the offline demo.

## What the numbers actually mean

These are the current measured results on the grouped **254-image test**. The detector improvement was evaluated on the same test after earlier results had been viewed, so treat the comparison as exploratory.

| Question | Result | Plain meaning |
| --- | --- | --- |
| Did the expanded detector place a matching box? | **248/254** (97.6%) | Finding the plate usually works on this test. |
| Did the complete photo-to-text system read every character correctly? | **71/254** (28.0%) | Exact reading is much harder than finding the plate. |
| How many were accepted automatically at the 10% validation target? | **75/254** (29.5% coverage) | The other 179 were not automatically accepted. |
| How many accepted readings were wrong? | **11/75** (14.7% observed error) | The validation target was **not achieved** on this test. |
| What happened with the stricter 2% and 5% targets? | **0/254** accepted automatically | The available evidence did not justify any automatic accepts under those rules. |

Do not describe 97.6% detector box matching as “97.6% plate reading accuracy.” Also, an accepted result can still be wrong. The 10% setting is a *threshold-selection target*, and its actual observed error here was 14.7%.

The dataset has some inaccurate XML labels, repeated video frames, foreign plates, and limited coverage of difficult conditions. The results describe this academic experiment; they are not a claim of reliable performance at every camera, road, or time of day. **OCR is the main current bottleneck.** The appropriate next research step is to curate better Indian plate labels, improve or train the reader, and evaluate again without using the test repeatedly to guide changes.

## What you can show in the faculty demo

1. Start with a **full vehicle photo** in the gallery. Point to the original photo, detected box, crop, and proposed text.
2. Point to **OCR model score** and **correctness estimate**. Explain that they answer different questions.
3. Point to **ACCEPT** or **REVIEW** and read the stated reason. Show how a person can correct an uncertain result.
4. Open the **Measured evidence** tab. State the 71/254 exact reading result and the 75 accepted / 11 wrong result honestly.
5. If asked where errors come from, compare full-photo mode with **Plate crop** mode on the same example. Show that a correct box can still lead to wrong characters.

Choose one example that works and one that fails before the presentation. A failure is useful evidence if you can explain which stage caused it. Do not imply that a gallery example is a separate test result.

## A short explanation you can say aloud

> “I started with a prototype that asked Gemini to read a number plate. For this project I built a local pipeline and measured each stage. First, a YOLO detector finds the plate in a full vehicle photo. Then FastPlateOCR reads the cropped plate. A small confidence model uses the OCR result and image information to estimate whether the whole plate is right. The app accepts only readings that pass a threshold chosen on separate validation data; the rest go to human review.
>
> On our grouped 254-image test, the expanded detector placed a matching box for 248 images, but the complete system read only 71 plate strings exactly. At the 10% validation target, it accepted 75 automatically, and 11 of those were wrong. So I would not deploy it unattended. The project shows a working offline system, measured limitations, and a clear direction for improving OCR.”

## Questions you may get

**“What is your contribution if YOLO and OCR already exist?”**  
“I put the parts together for vehicle photos, trained the detector on this data, compared readers, built a correctness estimate and review rule, and measured the full system. I am not claiming to have invented YOLO or a new OCR network.”

**“Why do you need two confidence numbers?”**  
“The OCR score comes from the reader. The correctness estimate is trained to predict whether *every character* in the plate is right. A strong OCR score can still accompany a one-character mistake.”

**“Does REVIEW mean the system failed completely?”**  
“No. It may have a useful proposed reading, but it is not confident enough to trust automatically. A person can inspect and correct it. If detection misses entirely, that is a full-image reading failure and also needs review.”

**“Could Gemini replace the whole pipeline?”**  
“It could provide a reading or a second opinion, but I would still need labeled tests and a rule for handling errors. The local pipeline makes each stage visible and works without an API key.”

**“Can you claim the system meets a 10% error limit?”**  
“No. The threshold was selected with that target on validation data, but the test showed 11 errors among 75 accepted readings, or 14.7%. I report the observed result.”

**“Does saving a correction improve tomorrow's predictions?”**  
“Not by itself. The correction is logged locally. It would need to be checked and included in a later, controlled training run.”

**“What would you improve next?”**  
“I would first audit and correct more labels, then improve OCR for small, blurry, and unusual Indian plates. I would use a fresh evaluation set to measure whether it actually helps.”

## Tiny glossary

| Word | Simple meaning |
| --- | --- |
| **Annotation / label** | The answer attached to an image: plate location and/or text. |
| **Detector** | The model that finds the plate's location. |
| **Crop** | The smaller image cut out around the plate. |
| **OCR** | The reader that turns plate pixels into letters and digits. |
| **Fine-tuning** | Adapting a pretrained model to our labeled examples. |
| **Calibration** | Checking and improving what a confidence number means. |
| **Threshold** | The minimum estimate required for automatic acceptance. |
| **Coverage** | The share of images accepted automatically. |
| **Accepted error** | The share of automatically accepted readings that are wrong. |
| **Test set** | Images reserved to measure the system after training and threshold choice. |

**If you only remember three things:** finding a plate is different from reading it; a high model score is not proof that the whole text is right; and uncertain readings must go to a person. For exact results and caveats, keep the [evaluation report](EVALUATION_REPORT.md) beside this guide.
