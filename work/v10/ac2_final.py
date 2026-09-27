"""Final India addressed no-candidate channel: 3-seed 'mid' LightGBM, validation gain per threshold, test accept parquet."""
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
tr = d.join(vq, on="q_row", how="anti"); va = d.join(vq, on="q_row", how="semi"); del d
X = tr.select(F).to_numpy().astype(np.float32); y = tr["label"].to_numpy(); XV = va.select(F).to_numpy().astype(np.float32)
ms = []
for seed in (1, 2, 3):
    m = lgb.train(dict(objective="binary", num_leaves=31, min_data_in_leaf=300, learning_rate=0.05, lambda_l2=5, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                       seed=seed, verbose=-1, num_threads=14), lgb.Dataset(X, y), 500)
    m.save_model(W+f"v10/ac2_mid_s{seed}.lgb"); ms.append(m)
pv = np.column_stack([m.predict(XV) for m in ms])
T = {"US": .75, "India": .8}
s1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gtv = load_ground_truth().join(s1.select("s1_id"), on="s1_id", how="semi")
v = pl.read_parquet(W+"valid_scores_stage2.parquet").join(s1, on="s1_id")
top = v.sort("p2", descending=True).unique("s23_id", keep="first")
base = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T)).select("s1_id","s23_id")
resc = pl.read_parquet(W+"v10/rescue_valid.parquet").sort("pr", descending=True).unique("q_row", keep="first").filter(pl.col("vs1") & (pl.col("pr") >= .9)).join(base, on="s23_id", how="anti")
base13 = pl.concat([base, resc.select("s1_id","s23_id")]).unique()
casc = pl.read_parquet(W+"cascade_valid_p001.parquet").select("s23_id").unique()
b0 = macro_f05(s1["s1_id"], base13, gtv, by=s1); print("base", {k: round(x, 5) for k, x in b0.items() if k.startswith("f05")})
for i in range(3):
    r = va.with_columns(pl.Series("pr", pv[:, i])).sort("pr", descending=True).unique("q_row", keep="first").filter(pl.col("vs1")).join(casc, on="s23_id", how="anti").join(base13, on="s23_id", how="anti")
    add = r.filter(pl.col("pr") >= .9); f = macro_f05(s1["s1_id"], pl.concat([base13, add.select("s1_id","s23_id")]), gtv, by=s1)["f05"]
    print("seed", i + 1, "t=.9", len(add), round(add["label"].mean(), 3), "gain %.5f" % (f - b0["f05"]))
va = va.with_columns(pl.Series("pr", pv.mean(1)))
va.select("q_row","s1_id","s23_id","label","vs1","pr").write_parquet(W+"v10/ac2_valid_mid.parquet")
r = va.sort("pr", descending=True).unique("q_row", keep="first").filter(pl.col("vs1")).join(casc, on="s23_id", how="anti").join(base13, on="s23_id", how="anti")
for t in (.6, .7, .8, .85, .9, .93, .95, .97):
    add = r.filter(pl.col("pr") >= t)
    mm = macro_f05(s1["s1_id"], pl.concat([base13, add.select("s1_id","s23_id")]), gtv, by=s1)
    print("ens", t, len(add), round(add["label"].mean(), 3), {k: round(x - b0[k], 5) for k, x in mm.items() if k.startswith("f05")}, flush=True)
del tr, va, X, XV
te = pl.read_parquet(W+"v10/ac2_test.parquet")
te = te.with_columns(pl.Series("pr", np.mean([m.predict(te.select(F).to_numpy().astype(np.float32)) for m in ms], 0)))
tt = te.sort("pr", descending=True).unique("q_row", keep="first").select("s1_id","s23_id","pr")
tt.filter(pl.col("pr") >= .6).write_parquet(W+"v10/ac2_test_top.parquet")
for t in (.6, .7, .8, .85, .9, .93, .95, .97):
    print("test", t, tt.filter(pl.col("pr") >= t).height)
