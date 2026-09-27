"""Paths and global settings."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("BER_DATA", ROOT / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK", ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT", ROOT / "output"))

SEED = 42
VALID_FRAC = 0.10  # fraction of train S1 entities held out for validation

for d in (WORK_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def is_valid_expr(col: str = "s1_id"):
    """Deterministic ~10% hash split of train S1 entities (validation holdout)."""
    import polars as pl
    return (pl.col(col).hash(seed=SEED) % 1000) < int(VALID_FRAC * 1000)
