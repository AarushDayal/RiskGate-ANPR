# RiskGate ANPR

An offline Indian number plate research prototype. It detects a plate in a vehicle photo, reads the crop with a local OCR model, estimates whether the whole reading is correct, and sends uncertain cases for human review. The [study guide](docs/STUDY_GUIDE.md) explains every stage in simple language; [evaluation report](docs/EVALUATION_REPORT.md) records actual results and limitations.

## Current measured result

On 254 grouped test images, the detector matched 237 plate boxes at IoU ≥ 0.5. The complete photo-to-text pipeline read 71/254 whole plates exactly (28.0%). At the threshold selected for a 10% validation error target, it accepted 75/254 automatically and 11 accepted readings were wrong (14.7% observed error). These are prototype results on XML labels that have only a sampled visual audit; do not present them as production accuracy. Strict 2% and 5% targets accepted no plates.

## Run the demo

The local environment already has its dependencies and generated model files:

```sh
./venv/bin/streamlit run app/streamlit_app.py
```

Open the local address Streamlit prints. The demo uses local models and makes no API calls. Choose **Full vehicle photo** for the detector-plus-OCR flow, or **Plate crop** to examine OCR separately. The evidence tab shows the test plots and detector results. Human corrections are saved locally in `results/review_corrections.csv`.

The [nine-slide presentation](RiskGate_Presentation.pdf) is ready to use; [presentation.html](presentation.html) supports arrow-key navigation. The local `weights/` folder contains the detector and OCR files needed for an offline demo on this machine.

`RiskGate_Submission.zip` is a portable copy of the demo, models, charts, report, and slides without dataset images or API keys. It is generated with `./venv/bin/python -m scripts.package_submission`. On another machine, create a Python virtual environment, install `requirements.txt`, and run the Streamlit command. Built-in dataset examples require the separate source dataset; uploaded images work without it.

## Rebuild the experiment

The source archive is the [Saisirishan Indian vehicle dataset](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset). Place its extracted folders under `data/indian/`, then run:

```sh
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m scripts.prepare_indian
./venv/bin/python -m scripts.detector prepare
./venv/bin/python -m scripts.baseline_v2
./venv/bin/python -m scripts.eval_router --fit
./venv/bin/python -m scripts.eval_router --final
./venv/bin/python -m scripts.detector train --epochs 12
./venv/bin/python -m scripts.eval_end_to_end
./venv/bin/python -m scripts.plot_results
make check
```

GPU training and GPU EasyOCR inference require a working CUDA installation. The project was run with an RTX 4060 Laptop GPU; FastPlateOCR uses local ONNX CPU inference. The detector training run was interrupted after three completed epochs; its best available checkpoint was copied to `weights/plate_detector.pt` and evaluated. The one-time test commands refuse to overwrite final results.

The source images, crops, generated weights, cache, and result files are Git-ignored. Do not redistribute the dataset. See [data notes](docs/DATA_NOTES.md) and [project status](docs/PROJECT_STATUS.md).
