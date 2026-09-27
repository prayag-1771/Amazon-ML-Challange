"""Stage-2 feature-set variants: CV F0.5 on validation + label-free test FP estimate from house-number shift symmetry."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "../work")
import numpy as np, polars as pl, lightgbm as lgb
import src.stage2 as s2
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
from symfix import prep, T
W = "../work/"
gt = load_ground_truth()
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
dv = s2.sib_feats(s2.struct_feats(s2.feats(s2.base("valid")), "train"), "train")
qf = dv.filter(pl.col("q_rank") == 1).select("s23_id", (pl.col("s1_id").hash(3) % 5).alias("fold"))
dv = dv.join(qf, on="s23_id").join(s1v, on="s1_id")
dt = s2.sib_feats(s2.struct_feats(s2.feats(s2.base("test")), "test"), "test")
tc, _ = prep("test_scores_stage2_v10.parquet", "test"); tc = tc.select("s23_id", "country", "dg", "up")
R = pl.read_parquet(W + "sym_R.parquet")
th = pl.col("country").replace_strict(T)
B = s2.F[:24]; S = ["nrel_c", "add_n", "drop_n"]
V = {"v9": B, "struct": B + S, "struct_sibtok": B + S + ["sib_add_min", "sib_add_max", "sib_drop_min", "sib_drop_max"],
     "struct_sibname": B + S + ["sib_name"], "v10": s2.F}
for nm, F in V.items():
    oof = np.zeros(len(dv), np.float32)
    for k in range(5):
        tr, te = (dv["fold"] != k).to_numpy(), (dv["fold"] == k).to_numpy()
        m = lgb.train(s2.P, lgb.Dataset(dv.filter(tr).select(F).to_numpy(), dv.filter(tr)["label"].to_numpy()), s2.R)
        oof[te] = m.predict(dv.filter(te).select(F).to_numpy(), num_threads=14)
    top = dv.with_columns(pl.Series("q", oof)).sort("q", descending=True).unique("s23_id", keep="first")
    mm = macro_f05(s1v["s1_id"], top.filter(pl.col("q") >= th).select("s1_id", "s23_id"), gt, by=s1v)
    m = lgb.train(s2.P, lgb.Dataset(dv.select(F).to_numpy(), dv["label"].to_numpy()), s2.R)
    m.save_model(W + f"stage2_var_{nm}.lgb")
    pt = m.predict(dt.select(F).to_numpy(), num_threads=14)
    tt = dt.select("s1_id", "s23_id").with_columns(pl.Series("q", pt))
    tt.write_parquet(W + f"test_scores_var_{nm}.parquet")
    top = tt.sort("q", descending=True).unique("s23_id", keep="first").join(tc, on="s23_id")
    a = top.filter(pl.col("dg").is_not_null() & (pl.col("q") >= th)).group_by("country", "dg").agg(
        pl.col("up").sum().alias("au"), (~pl.col("up")).sum().alias("ad")).join(R, on=["country", "dg"])
    ex = a.group_by("country").agg((pl.col("au") - pl.col("R") * pl.col("ad")).sum().round(0).alias("excess"), pl.col("au").sum()).sort("country").rows()
    acc = top.group_by("country").agg((pl.col("q") >= th).sum()).sort("country").rows()
    print(f"{nm:15s} CV {mm['f05']:.5f} (US {mm['f05_US']:.5f} IN {mm['f05_India']:.5f})  test excess-up {ex}  acc {acc}", flush=True)
