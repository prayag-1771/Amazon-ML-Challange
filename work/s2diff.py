"""Stage-2 CV: add the exact signed first-house-number difference (and distractor-offset flag)."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl, lightgbm as lgb
import src.stage2 as S
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W = "../work/"
s1v = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt = load_ground_truth()
d = S.ce_hard_feats(S.sib_feats(S.struct_feats(S.feats(S.base("valid")), "train"), "train"), "valid")
s1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","nums"]).rename({"entity_id":"s1_id","nums":"u1"})
q = pl.concat([pl.read_parquet(W+f"norm_train_s{i}.parquet", columns=["entity_id","nums"]) for i in (2,3)]).rename({"entity_id":"s23_id","nums":"uq"})
x = d.select("s1_id","s23_id").join(s1,on="s1_id",how="left").join(q,on="s23_id",how="left")
a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False); b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
df = b - a
x = x.select("s1_id","s23_id", pl.when(df.abs() <= 30).then(df).otherwise(pl.when(df > 0).then(99).otherwise(-99)).alias("ndiff"),
             df.is_in([3,4,5,7,9,11,13,21]).cast(pl.Int8).alias("nd3"),
             (b.cast(pl.Float64)/a.cast(pl.Float64)).alias("nratio"))
d = d.join(x, on=["s1_id","s23_id"], how="left")
qf = d.filter(pl.col("q_rank")==1).select("s23_id", (pl.col("s1_id").hash(3) % 5).alias("fold"))
d = d.join(qf, on="s23_id").join(s1v, on="s1_id")
y = d["label"].to_numpy(); fold = d["fold"].to_numpy()
th = pl.col("country").replace_strict(S.T, default=0.75)
def cv(F, name):
    X = d.select(F).to_numpy().astype(np.float32); oof = np.zeros(len(d), np.float32)
    for k in range(5):
        tr, te = fold != k, fold == k
        oof[te] = lgb.train(S.P, lgb.Dataset(X[tr], y[tr]), S.R).predict(X[te], num_threads=14)
    top = d.select("s1_id","s23_id","country").with_columns(pl.Series("p2", oof)).sort("p2", descending=True).unique("s23_id", keep="first")
    res = []
    for dt in (-0.05, 0, 0.05):
        m = macro_f05(s1v["s1_id"], top.filter(pl.col("p2") >= th + dt).select("s1_id","s23_id"), gt, by=s1v)
        res.append(f"dt{dt:+.2f} {m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f})")
    print(name, " | ".join(res), flush=True)
cv(S.F, "base   ")
cv(S.F + ["ndiff"], "+ndiff ")
cv(S.F + ["ndiff","nd3","nratio"], "+ndiff3")
