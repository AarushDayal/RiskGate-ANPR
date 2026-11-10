"""Make a portable demo bundle with gallery photos but no bulk dataset or API keys."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from riskgate import config

ROOT = config.ROOT
OUTPUT = ROOT / "RiskGate_Submission.zip"
SINGLE = ["README.md", "requirements.txt", "Makefile", "presentation.html",
          "RiskGate_Presentation.pdf", "calibration.json", ".streamlit/config.toml"]
FOLDERS = ["app", "riskgate", "scripts", "docs", "weights/fastplate", "data/sample_gallery"]
RESULTS = ["selected_reader.json", "final_test.json", "end_to_end_test.json",
           "end_to_end_extra.json", "detector_test.json", "detector_comparison.json",
           "baseline_v2_summary.txt", "risk_coverage.png", "reliability.png", "audit_issues.csv"]


def main() -> None:
    files = [ROOT / name for name in SINGLE]
    for folder in FOLDERS:
        files += [path for path in (ROOT / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    files += [ROOT / "weights/plate_detector.pt"]
    files += [ROOT / "results" / name for name in RESULTS]
    missing = [str(path.relative_to(ROOT)) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required bundle files: {missing}")
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MiB, {len(files)} files)")


if __name__ == "__main__":
    main()
