"""Regularized retrain of ac2 model; report val precision/gain per threshold for several configs."""
import sys; sys.path.insert(0, "../business_entity_resolution")
import numpy as np, polars as pl, lightgbm as lgb
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W = "../work/"
F = ["asim","nsim","comb","rk_asim","rk_nsim","rk_comb","num_sh","num_q","num_1","w_sh","w_q","w_1","key_eq","eq","leq","num_fq","w_fq","r","ts","jw","kr","len1","lenq",
     "n_c","s1_nq","asim_gap","nsim_gap","comb_gap","kr_gap","num_fq_gap","w_fq_gap","comb_gap2","comb_s1gap"]
d = pl.read_parquet(W+"v10/ac2_train.parquet")
gt = pl.read_parquet(W+"gt_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("label"))
d = d.join(gt, on=["s1_id","s23_id"], how="left").with_columns(pl.col("label").fill_null(0), is_valid_expr("s1_id").alias("vs1"))
vq = d.filter(pl.col("vs1")).select("q_row").unique()
tr = d.join(vq, on="q_row", how="anti"); va = d.join(vq, on="q_row", how="semi")
T = {"US": .75, "India": .8}
s1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gtv = load_ground_truth().join(s1.select("s1_id"), on="s1_id", how="semi")
v = pl.read_parquet(W+"valid_scores_stage2.parquet").join(s1, on="s1_id")
top = v.sort("p2", descending=True).unique("s23_id", keep="first")
base = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T)).select("s1_id","s23_id")
resc = pl.read_parquet(W+"v10/rescue_valid.parquet").sort("pr", descending=True).unique("q_row", keep="first").filter(pl.col("vs1") & (pl.col("pr") >= .9)).join(base, on="s23_id", how="anti")
base13 = pl.concat([base, resc.select("s1_id","s23_id")]).unique()
casc = pl.read_parquet(W+"cascade_valid_p001.parquet").select("s23_id").unique()
b0 = macro_f05(s1["s1_id"], base13, gtv, by=s1)["f05"]
X = tr.select(F).to_numpy().astype(np.float32); y = tr["label"].to_numpy(); XV = va.select(F).to_numpy().astype(np.float32)
for name, prm, n in [("small", dict(num_leaves=15, min_data_in_leaf=500, learning_rate=0.05, lambda_l2=10), 300),
                     ("mid", dict(num_leaves=31, min_data_in_leaf=300, learning_rate=0.05, lambda_l2=5), 500)]:
    m = lgb.train(dict(objective="binary", feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=14, **prm), lgb.Dataset(X, y), n)
    p = m.predict(XV)
    r = va.with_columns(pl.Series("pr", p)).sort("pr", descending=True).unique("q_row", keep="first").filter(pl.col("vs1")).join(casc, on="s23_id", how="anti").join(base13, on="s23_id", how="anti")
    ptr = m.predict(X); rt = tr.with_columns(pl.Series("pr", ptr)).sort("pr", descending=True).unique("q_row", keep="first")
    for t in (.9, .95, .98, .99):
        add = r.filter(pl.col("pr") >= t); at = rt.filter(pl.col("pr") >= t)
        f = macro_f05(s1["s1_id"], pl.concat([base13, add.select("s1_id","s23_id")]), gtv, by=s1)["f05"]
        print(name, t, "val", len(add), round(add["label"].mean(), 3), "gain %.5f" % (f - b0), "| train-in-sample", len(at), round(at["label"].mean(), 3), flush=True)
