import sys; sys.path.insert(0, "."); sys.path.insert(0, "../work")
import numpy as np, polars as pl, lightgbm as lgb
from stage2 import base, feats, F, P, R, W
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
d = feats(base("valid"))
sch = pl.read_parquet_schema(W + "valid_feats.parquet")
extra = [c for c, t in sch.items() if c not in d.columns and c not in ("q_row", "s1_row", "label") and t.is_numeric()]
d = d.join(pl.read_parquet(W + "valid_feats.parquet", columns=["s1_id", "s23_id"] + extra), on=["s1_id", "s23_id"])
qf = d.filter(pl.col("q_rank") == 1).select("s23_id", (pl.col("s1_id").hash(3) % 5).alias("fold"))
d = d.join(qf, on="s23_id")
def cv(FF, PP, RR):
    oof = np.zeros(len(d), np.float32)
    for k in range(5):
        tr, te = d["fold"] != k, d["fold"] == k
        m = lgb.train(PP, lgb.Dataset(d.filter(tr).select(FF).to_numpy(), d.filter(tr)["label"].to_numpy()), RR)
        oof[te.to_numpy()] = m.predict(d.filter(te).select(FF).to_numpy(), num_threads=14)
    top = d.with_columns(pl.Series("p2", oof)).sort("p2", descending=True).unique("s23_id", keep="first")
    out = []
    for t in [0.65, 0.7, 0.75, 0.8]:
        m2 = macro_f05(s1v["s1_id"], top.filter(pl.col("p2") >= t).select("s1_id", "s23_id"), gt, by=s1v)
        out.append(f"t={t} {m2['f05']:.5f} (US {m2['f05_US']:.5f} IN {m2['f05_India']:.5f})")
    return " | ".join(out)
print("rows", len(d), "extra", len(extra), flush=True)
print("base F      ", cv(F, P, R), flush=True)
print("F+all feats ", cv(F + extra, P, R), flush=True)
P2 = dict(P, learning_rate=0.03, num_leaves=63)
print("F+all 63l800", cv(F + extra, P2, 800), flush=True)
