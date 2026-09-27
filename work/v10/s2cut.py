"""Candidate-set reduction: v11 stage-2 CV when the candidate set is cut to first-stage p >= PCUT (features recomputed
on the cut set, as they would be at inference). Also reports the resulting test pairs per S1."""
import sys
sys.path.insert(0, ".")
import numpy as np, polars as pl, lightgbm as lgb
from src import stage2 as S
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W = S.W
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
tp = pl.read_parquet(W + "test_scores_stage2_v11.parquet", columns=["p"])["p"]
nt1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id"]).height
B = S.base("valid")
ntrue = gt.height if hasattr(gt, "height") else None
for PCUT in [float(x) for x in sys.argv[1:]]:
    b = B.filter(pl.col("p") >= PCUT) if PCUT > 0 else B
    d = S.ce_hard_feats(S.sib_feats(S.struct_feats(S.feats(b), "train"), "train"), "valid")
    qf = d.filter(pl.col("q_rank") == 1).select("s23_id", (pl.col("s1_id").hash(3) % 5).alias("fold"))
    d = d.join(qf, on="s23_id")
    oof = np.zeros(len(d), np.float32)
    for k in range(5):
        tr, te = d["fold"] != k, d["fold"] == k
        m = lgb.train(S.P, lgb.Dataset(d.filter(tr).select(S.F).to_numpy(), d.filter(tr)["label"].to_numpy()), S.R)
        oof[te.to_numpy()] = m.predict(d.filter(te).select(S.F).to_numpy(), num_threads=14)
    top = d.with_columns(pl.Series("p2", oof)).sort("p2", descending=True).unique("s23_id", keep="first").join(s1v, on="s1_id")
    th = pl.col("country").replace_strict(S.T, default=0.75)
    mm = macro_f05(s1v["s1_id"], top.filter(pl.col("p2") >= th).select("s1_id", "s23_id"), gt, by=s1v)
    nte = int((tp >= PCUT).sum())
    print(f"PCUT={PCUT:g}: val {len(b)/len(s1v):.2f}/S1 true kept {int(b['label'].sum()):,}  F {mm['f05']:.5f} (US {mm['f05_US']:.5f} IN {mm['f05_India']:.5f})  test {nte:,} pairs {nte/nt1:.3f}/S1", flush=True)
