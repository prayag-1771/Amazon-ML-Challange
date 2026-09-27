"""v16: re-score the India addressed channel's test pairs with the retrained models (india_addr16_s{1,2,3}.lgb).

The pairs file india_addr_test_pairs_v15c.parquet is the v15 channel's test candidate set: its v15-model scores
reproduce v15's 5,505 accepted pairs exactly. The v16 models were trained by work/ia16train.py, which is
src.india_addr's training with the current code. Validation (work/prsweep16.py, unbiased holdout of every record
touching a validation S1): cutoff 0.6 -> +0.00015, 0.7 -> +0.00013 (precision 0.949), 0.9 -> +0.00002.

  python v16/score_india16.py <cutoff>      (from work/)  -> v16/india16_top1.parquet, v16/india16_accept.parquet
"""
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parent / "business_entity_resolution"))
import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from src import india_addr as ia

W = str(WORK) + "/"
cutoff = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
models = [lgb.Booster(model_file=W + f"india_addr16_s{s}.lgb") for s in ia.SEEDS]
pf = pq.ParquetFile(W + "india_addr_test_pairs_v15c.parquet")
parts = []
for i, b in enumerate(pf.iter_batches(batch_size=2_000_000, columns=["s1_id", "s23_id"] + ia.F)):
    d = pl.from_arrow(b)
    X = d.select(ia.F).to_numpy().astype(np.float32)
    parts.append(d.select("s1_id", "s23_id").with_columns(pl.Series("pr", np.mean([m.predict(X, num_threads=8) for m in models], axis=0))))
    print(f"  batch {i}: {sum(p.height for p in parts):,} pairs scored", flush=True)
d = pl.concat(parts)
top = d.sort(["pr", "s1_id"], descending=[True, False]).unique("s23_id", keep="first")
top.write_parquet(WORK / "v16" / "india16_top1.parquet")
print("queries", top.height, {t: top.filter(pl.col("pr") >= t).height for t in (0.5, 0.6, 0.7, 0.8, 0.9)})
acc = top.filter(pl.col("pr") >= cutoff)
acc.write_parquet(WORK / "v16" / "india16_accept.parquet")
old = pl.read_parquet(W + "india_addr_test_accept.parquet")
print(f"accepted at {cutoff}: {acc.height:,} (v15 had {old.height:,}; overlap {acc.join(old, on=['s1_id', 's23_id'], how='semi').height:,}, "
      f"v15 pairs dropped {old.join(acc, on=['s1_id', 's23_id'], how='anti').height:,})")
