"""Orphan simulation: drop a fraction f of validation S1 entities from the candidate pool (their true
S2/S3 records stay as queries with no correct S1), score the remaining S1 entities."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
d = pl.read_parquet(W + "valid_scores_stage2.parquet")
for f in (0.0, 0.1, 0.2, 0.3):
    drop = s1v.filter((pl.col("s1_id").hash(11) % 1000) < int(f * 1000))
    keep = s1v.join(drop, on="s1_id", how="anti")
    dd = d.join(drop.select("s1_id"), on="s1_id", how="anti")
    q_all = d["s23_id"].n_unique(); q_pos = dd.join(gt.join(keep.select("s1_id"), on="s1_id"), on=["s1_id", "s23_id"], how="semi")["s23_id"].n_unique()
    print(f"f={f} negshare~{1 - q_pos / q_all:.3f}")
    for col in ("p", "p2"):
        top = dd.sort(col, descending=True).unique("s23_id", keep="first")
        r = []
        for t in (0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            m = macro_f05(keep["s1_id"], top.filter(pl.col(col) >= t).select("s1_id", "s23_id"), gt, by=keep)
            r.append(f"{t}:{m['f05']:.5f}/P{m['precision_micro']:.4f}")
        print(f"  {col} " + " ".join(r), flush=True)
