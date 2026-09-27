"""Stage-2 CV experiments on val: baseline v10 features, + hard-CE feature, + candidate cut (p >= PCUT)."""
import sys
sys.path.insert(0, ".")
import numpy as np, polars as pl, lightgbm as lgb
from src import stage2 as S
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W = S.W
tag = sys.argv[1] if len(sys.argv) > 1 else "s1"
PCUT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
b = S.base("valid")
n0 = len(b)
if PCUT > 0:
    b = b.filter(pl.col("p") >= PCUT)
print(f"pairs {n0:,} -> {len(b):,} ({len(b)/len(s1v):.2f}/S1), true kept {b['label'].sum():,}", flush=True)
d = S.sib_feats(S.struct_feats(S.feats(b), "train"), "train")
vf = pl.read_parquet(W + "valid_feats.parquet", columns=["q_row", "s1_row", "s1_id", "s23_id"])
h = pl.read_parquet(W + f"ce_hard_{tag}_valid.parquet")
d = d.join(vf.join(h, on=["q_row", "s1_row"]).select("s1_id", "s23_id", "ce_h"), on=["s1_id", "s23_id"], how="left")
d = d.with_columns(pl.col("ce_h").rank("ordinal", descending=True).over("s23_id").alias("ce_h_rank"),
                   (pl.col("ce_h") - pl.col("ce_h").max().over("s23_id")).alias("ce_h_gap"))
print("ce_h null share", d["ce_h"].is_null().mean())
qf = d.filter(pl.col("q_rank") == 1).select("s23_id", (pl.col("s1_id").hash(3) % 5).alias("fold"))
d = d.join(qf, on="s23_id")
def cv(F):
    oof = np.zeros(len(d), np.float32)
    for k in range(5):
        tr, te = d["fold"] != k, d["fold"] == k
        m = lgb.train(S.P, lgb.Dataset(d.filter(tr).select(F).to_numpy(), d.filter(tr)["label"].to_numpy()), S.R)
        oof[te.to_numpy()] = m.predict(d.filter(te).select(F).to_numpy(), num_threads=14)
    top = d.with_columns(pl.Series("p2", oof)).sort("p2", descending=True).unique("s23_id", keep="first")
    res = []
    for t in [0.7, 0.75, 0.8]:
        m = macro_f05(s1v["s1_id"], top.filter(pl.col("p2") >= t).select("s1_id", "s23_id"), gt, by=s1v)
        res.append(f"t={t} {m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f})")
    return "  ".join(res)
print("v10       ", cv(S.F), flush=True)
print("v10+ce_h  ", cv(S.F + ["ce_h", "ce_h_rank", "ce_h_gap"]), flush=True)
