import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "ZARA_jackets_men.csv"
INDEXES_DIR = ROOT / "indexes"


def catalog_path():
    # CATALOG_PATH lets users point the app and evaluation at their own CSV.
    return Path(os.environ.get("CATALOG_PATH") or DEFAULT_CATALOG).expanduser().resolve()
