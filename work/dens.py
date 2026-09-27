"""Validation with test-like distractor density: duplicate pure-distractor queries (no ground truth at all)."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05
W="../work/"
gt = pl.read_parquet(W+"gt_pairs.parquet")
s1v = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country"]).rename({"entity_id":"s1_id"}).filter(is_valid_expr("s1_id"))
sc = pl.read_parquet(W+"valid_scores_lgb_v6k.parquet")
top = sc.sort("p", descending=True).unique("s23_id", keep="first").select("s1_id","s23_id","p")
dq = top.join(gt.select("s23_id"), on="s23_id", how="anti")
print("distractor queries in val top:", len(dq), "of", len(top))
rng = np.random.default_rng(0)
for mult in (1.0, 1.5, 1.9, 2.5):
    extra = int(round((mult-1)*len(dq)))
    parts = [top]
    k = 0
    while extra > 0:
        take = dq.sample(min(extra, len(dq)), seed=k).with_columns((pl.col("s23_id")+f"_dup{k}").alias("s23_id"))
        parts.append(take); extra -= len(take); k += 1
    tt = pl.concat(parts)
    res = []
    for t in np.round(np.arange(0.5, 0.99, 0.05), 2).tolist() + [0.97, 0.98, 0.99]:
        m = macro_f05(s1v["s1_id"], tt.filter(pl.col("p") >= t), gt, by=s1v)
        res.append((t, round(m["f05"], 5), round(m["precision_micro"], 5), round(m["recall_micro"], 5)))
    best = max(res, key=lambda r: r[1])
    print(f"mult {mult}: best {best}  | " + " ".join(f"{r[0]}:{r[1]}" for r in res), flush=True)
