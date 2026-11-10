import torch

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/legacy/datacluster"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "Annotations"
SPLITS_JSON = ROOT / "archive/splits.json"
SPLITS_V2 = ROOT / "data/splits_v2.json"
CALIBRATION_MODEL = ROOT / "weights/calibration.json"
DRAFTS = ROOT / "data/labels/draft_labels.csv"
YOLO_START = ROOT / "weights/yolo11n.pt"
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
