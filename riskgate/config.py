import torch

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datacluster"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "Annotations"
SPLITS_JSON = ROOT / "splits.json"
RESULTS_DIR = ROOT / "results"

SEED = 0
MIN_CROP_H = 0
SPLIT_FRACTIONS = (0.4, 0.3, 0.3)  # calib / val / test, split by IMAGE not by plate
READER_ALLOWLIST = True
READER_UPSCALE = False
GEMINI_MODEL = "gemini-2.5-flash"


def device() -> str:
    d = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"riskgate device: {d}")
    return d
