"""CE-only assignment on val (threshold sweep) + France disagreement vs v9 on test."""
import sys
sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
v = pl.read_parquet(W + "valid_feats.parquet", columns=["s1_id", "s23_id", "ce_logit"])
top = v.sort("ce_logit", descending=True).unique("s23_id", keep="first")
for t in (0, 1, 2, 3, 4, 5):
    r = macro_f05(s1v["s1_id"], top.filter(pl.col("ce_logit") >= t).select("s1_id", "s23_id"), gt, by=s1v)
    print("val CE-only t", t, round(r["f05"], 5), round(r["f05_US"], 5), round(r["f05_India"], 5), "P", round(r["precision_micro"], 5), "R", round(r["recall_micro"], 5), flush=True)
