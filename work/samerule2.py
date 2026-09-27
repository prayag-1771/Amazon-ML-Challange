"""Exact-address rule gated by label-free token rel (xq_relmin): validation F0.5 and test additions."""
import sys
sys.path.insert(0, ".")
import polars as pl
from src.metric import macro_f05
from src.config import is_valid_expr
W = "../work/"
T = 0.75
gt = pl.read_parquet(W + "gt_pairs.parquet")
TF = ["xq_n", "xq_relmin", "x1_n", "x1_relmin"]
def prep(split, sc, feats):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "addr_core", "country"]).rename({"entity_id": "s1_id", "addr_core": "a1"})
    s1 = s1.join(s1.filter(pl.col("a1") != "").group_by("country", "a1").len("a1_n"), on=["country", "a1"], how="left")
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "addr_core"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "addr_core": "aq"})
    top = sc.sort("p", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id").join(feats, on=["s1_id", "s23_id"], how="left")
    return top.with_columns(((pl.col("a1") != "") & (pl.col("a1") == pl.col("aq")) & (pl.col("a1_n") == 1)).alias("same1"))
vf = pl.read_parquet(W + "valid_feats.parquet", columns=["s1_id", "s23_id"] + TF)
top = prep("train", pl.read_parquet(W + "valid_scores_lgb_v5cf.parquet").select("s1_id", "s23_id", "p"), vf)
tf = pl.read_parquet(W + "test_feats.parquet", columns=["s1_id", "s23_id"] + TF)
tt = prep("test", pl.read_parquet(W + "test_scores_lgb_v5cf.parquet"), tf)
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
def score(sel):
    m = macro_f05(s1v["s1_id"], sel, gt, by=s1v)
    return round(m["f05"], 5), round(m["f05_US"], 5), round(m["f05_India"], 5)
print("base", score(top.filter(pl.col("p") >= T)))
print("xq_relmin describe (test same1 unassigned):"); print(tt.filter(pl.col("same1") & (pl.col("p") < T)).group_by("country").agg(pl.len(), pl.col("xq_relmin").quantile(0.1).alias("q10"), pl.col("xq_relmin").median().alias("med"), pl.col("x1_relmin").median().alias("x1med")))
for rmin in (-99, 0.2, 0.3, 0.4):
    for tl in (0.01, 0.05, 0.1, 0.3):
        cond = pl.col("same1") & (pl.col("p") >= tl) & ((pl.col("xq_relmin") >= rmin) | (pl.col("xq_n") == 0))
        sel = top.filter((pl.col("p") >= T) | cond)
        add = tt.filter(cond & (pl.col("p") < T)).group_by("country").len().sort("country").rows()
        print(f"rmin {rmin} tl {tl}", score(sel), "valid_added", sel.height - top.filter(pl.col("p") >= T).height, "test_added", add, flush=True)
