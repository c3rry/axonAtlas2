from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

for p in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)