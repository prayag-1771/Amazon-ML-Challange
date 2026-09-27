"""Generate Submission v8: Model ensemble (v5cf + v6k) with per-country calibrated decision thresholding."""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import polars as pl
from src.io_utils import load_source, write_id_lists
from src.config import OUTPUT_DIR, WORK_DIR

t0 = time.time()
print("Loading inputs...")
s1_test = load_source("test", 1)
s1_ids = s1_test["entity_id"].to_list()
s1_country = pl.read_parquet(WORK_DIR / "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})

a = pl.read_parquet(WORK_DIR / "test_scores_lgb_v5cf.parquet").rename({"p": "pa"})
b = pl.read_parquet(WORK_DIR / "test_scores_lgb_v6k.parquet").rename({"p": "pb"})

print(f"Joining predictions and applying per-country thresholds (US=0.70, India=0.80, France=0.70)...")
d = a.join(b, on=["s1_id", "s23_id"]).with_columns(((pl.col("pa") + pl.col("pb")) / 2.0).alias("p")).join(s1_country, on="s1_id")

# Save ensemble test scores
d.select("s1_id", "s23_id", "p").write_parquet(WORK_DIR / "test_scores_ens_v8.parquet")

matches = d.filter(
    ((pl.col("country") == "US") & (pl.col("p") >= 0.70)) |
    ((pl.col("country") == "India") & (pl.col("p") >= 0.80)) |
    ((pl.col("country") == "France") & (pl.col("p") >= 0.70))
).sort("p", descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id", "p")

print(f"Total matched pairs: {len(matches):,}")
counts = matches.join(s1_country, on="s1_id")["country"].value_counts()
print("Per-country match counts:\n", counts)

out_path = OUTPUT_DIR / "matching_results.tsv"
print(f"Writing {out_path}...")
write_id_lists(out_path, s1_ids, matches, "matched_entity_ids")
print(f"Done in {time.time() - t0:.1f}s")
