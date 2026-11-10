# RiskGate-ANPR: Part 1 walkthrough (Hinglish)

**Historical note:** This explains the early 25-plate EasyOCR phase. It is not the final system state. See [PROJECT_STATUS.md](PROJECT_STATUS.md), [EVALUATION_REPORT.md](EVALUATION_REPORT.md), and [STUDY_GUIDE.md](STUDY_GUIDE.md) for the current detector, OCR comparison, calibration, and demo.

## Har file kya karta hai (1 line each)
- `riskgate/contracts.py` — `PlateSample` (image+bbox+truth) aur `EngineResult` (text/conf/latency/error) dataclasses; engine error return karta hai, raise nahi.
- `riskgate/config.py` — sab paths, SEED=0, MIN_CROP_H, SPLIT_FRACTIONS, aur device() (prints `cuda`).
- `riskgate/data.py` — XMLs padh ke PlateSamples banata hai (text missing → skip, bbox clip), aur saari 47 images ke names ko seed se shuffle karke calib/val/test = 19/14/14 split karke `archive/splits.json` likhta hai (assert: koi image do splits me nahi).
- `riskgate/metrics.py` — normalize, exact_match, CER (edit distance / len(truth)), ECE (10 bins), risk-coverage points.
- `riskgate/grammar.py` — Indian plate classify: VALID / NONSTANDARD / INVALID, sirf classify, kabhi "correct" nahi karta.
- `riskgate/engines/base.py` — Engine protocol: read(image) -> EngineResult, kabhi raise nahi.
- `riskgate/engines/easyocr_engine.py` — real engine, GPU pe EasyOCR; allowlist + 2x upscale flags (default OFF), fail hua to error result.
- `riskgate/engines/mock.py` — test ke liye fixed-result engine, koi model/network nahi.
- `tests/` — 12 tests, har metric ka hand-computed expected value comment me.
- `Makefile` — `make check` = pytest + 10-image smoke run.
- `scripts/run_eval.py` — 3 variants (baseline / allowlist / allowlist+upscale) ko table + `results/baseline_easyocr.csv` (image, plate index, truth, pred, conf, grammar_class, ok) me likhta hai; smoke run CSV overwrite nahi karta.
- `archive/eval_easyocr.py` — old prototype kept for reference; its CSV output now stays in `archive/`.

## Ek plate ka data flow (6 steps)
1. `data.py` `Annotations/*.xml` parse karta hai → `<attribute>` me `number_plate_text` mila to PlateSample ban gaya (bbox image se clip).
2. `make_splits()` image names shuffle (seed 0) karke image ko calib/val/test me daalta hai → `archive/splits.json`.
3. Eval me image `cv2.imread` → bbox se crop `data.crop()`.
4. Crop `engine.read()` ko jata hai → EngineResult(text, raw_conf, latency, error).
5. `metrics.normalize()` pred+truth dono pe → `exact_match`, `cer`.
6. `grammar.classify()` pred pe → per-plate row CSV me + summary table (exact_match, mean CER, mean conf correct/wrong, grammar-valid correct/wrong).

## CER > 1 ka cause (5 concrete pairs se, purane eval_easyocr.csv se)
Purane code me CER = edit distance / **len(pred)** tha (len(truth) nahi). Jab pred khali tha:
- `image_0028 obj1`: truth=`KL34A465` (8 chars), pred=`''` → distance=8, len(pred)=1 → old CER = **8.0**
- `image_0028 obj5`: truth=`KL03S6894` (9 chars), pred=`''` → **9.0**
- `image_0033`: truth=`MP07L7524`, pred=`MPO7SL7524` → old 0.2 len(pred)=9 pe, new 0.222 len(truth)=8 pe (true value)
- `image_0047`: truth=`KA01AJ7533`, pred=`KA0IAJ7533` → 1 sub / 10 = 0.1 (dono sahi is case me, kyunki length same thi)
- → toh mean CER khisak ke > 1.0 chala gaya tha. Fix: CER = distance / len(truth). Ab max legitimate value 1.0 (empty pred) hai; > 1.0 sirf tab jab pred truth se bahut lamba junk ho.

## Comparison table (25 labelled plates, all images, GPU=cuda)
| variant | exact_match | mean CER | mean conf correct | mean conf wrong | grammar-valid correct | grammar-valid wrong |
|---|---|---|---|---|---|---|
| baseline | 0.16 | 0.417 | 0.497 | 0.467 | 1.0 | 0.095 |
| allowlist | 0.16 | 0.420 | 0.78 | 0.548 | 1.0 | 0.095 |
| allowlist+upscale | 0.12 | 0.441 | 0.891 | 0.543 | 1.0 | 0.0 |

**Winner: allowlist.** Accuracy same 0.16, CER ~same, lekin correct vs wrong ka confidence gap bohot bada ho gaya (0.497/0.467 → 0.78/0.548) — Part 2 calibration ke liye yahi cheez chahiye. Upscale: accuracy pichh gayi aur CER badha, lekin grammar-valid wrong = 0.0 (har galat read grammar-check pe pakda jayega) — routing ke liye useful signal. Per-plate detail: `results/baseline_easyocr.csv`.

## Part 2 se pehle 3 sawaal (khud apne words me jawab do)
1. Hum split IMAGE ke hisaab se karte hain, plate ke hisaab se kyun nahi? Same image ke do plates ko alag-alag split me daalne se kya problem hoti hai?
2. Allowlist ke baad conf gap bada hua lekin exact_match same raha — kya matlab hai ki "sahi par low conf" vs "galat par high conf" dono abhi bhi exist karte hain? Isse router ka decision boundary kahan baithna chahiye?
3. Ek 0.989 conf ka galat read (image_0041: TN58AP5280 → 58APTN5280) system ko user tak pahunch jaye toh cost kya hogi? Grammar-check ne use INVALID kar diya tha (NONSTANDARD/INVALID se shayad ESCALATE route bacha le).

## Unsure / skipped
- **NONSTANDARD vs INVALID boundary ambiguous thi** — spec ne sirf VALID ka format diya. Meri heuristic: ≥6 alnum chars + letters aur digits dono = NONSTANDARD, warna INVALID (empty, digit-only, chhota garbage). Agar spec clear kare toh isko change karna easy hai (`grammar.py` ek regex file hai).
- Dataset ke kuch truths (jaise `KL34A465`) 3 trailing digits wale hain — strict 4-digit standard me ye VALID nahi, isliye wo truths khud NONSTANDARD classify hote. Ye data/annotation ka issue lagta hai, labels ko haath nahi lagaya (rule 4).
- Gemini/network calls ya calibration/router/koi Part-2 cheez nahi banayi (rule 6 per).
- EasyOCR `+upscale` flag easyocr ke apne argument ke bajaye `cv2.resize(..., 2x)` se implement kiya — same effect, transparent.
