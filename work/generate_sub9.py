"""Generate Submission v9: stage-2 collective re-ranker over the v5cf + v6k ensemble scores.
Stage 2 (work/stage2.py) is a LightGBM trained on the validation queries, where first-stage scores are
out-of-sample. It adds per-query and per-S1 aggregates of first-stage scores (how many other queries
confidently claim this S1, from the same or the other source; competing S1's support)."""
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "../work")
import lightgbm as lgb
import polars as pl
from src.io_utils import load_source, write_id_lists
from src.config import OUTPUT_DIR, WORK_DIR
from stage2 import base, feats, F

T = {"US": 0.75, "India": 0.80, "France": 0.75}
t0 = time.time()
s1_ids = load_source("test", 1)["entity_id"].to_list()
s1_country = pl.read_parquet(WORK_DIR / "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
d = feats(base("test"))
assert d["ce_logit"].null_count() == 0
m = lgb.Booster(model_file=str(WORK_DIR / "stage2.lgb"))
d = d.with_columns(pl.Series("p2", m.predict(d.select(F).to_numpy(), num_threads=14))).join(s1_country, on="s1_id")
d.select("s1_id", "s23_id", "p", "p2").write_parquet(WORK_DIR / "test_scores_stage2_v9.parquet")
top = d.sort("p2", descending=True).unique("s23_id", keep="first")
thr = pl.col("country").replace_strict(T, default=0.75)
matches = top.filter(pl.col("p2") >= thr).select("s1_id", "s23_id", "country")
print(f"matched pairs {len(matches):,}", matches["country"].value_counts().sort("country"))
print("per S1:", {c: round(matches.filter(pl.col("country") == c).height / s1_country.filter(pl.col("country") == c).height, 3) for c in T})
write_id_lists(OUTPUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
print(f"done {time.time() - t0:.0f}s")
