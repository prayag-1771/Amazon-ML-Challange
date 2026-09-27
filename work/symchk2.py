import sys; sys.path.insert(0, "."); sys.path.insert(0, "../work")
import polars as pl
from symfix import prep, fix, K, T
R = pl.read_parquet("../work/sym_R.parquet")
t, _ = prep("test_scores_stage2_v10.parquet", "test")
th = pl.col("country").replace_strict(T)
for sl, z in ((1.0, 2.0),):
    tf, g = fix(t, R, sl, z)
    x = tf.filter(pl.col("dg").is_not_null() & pl.col("up") & (pl.col("p2") > 0.02))
    a = x.group_by(K).agg((pl.col("p2") >= th).sum().alias("acc0"), (pl.col("p3") >= th).sum().alias("acc1"))
    out = g.select(K + ["nu", "nd", "R", "tgt"]).join(a, on=K).with_columns((pl.col("R") * pl.col("nd")).round(0).alias("est")).sort("acc0", descending=True)
    with pl.Config(tbl_rows=60, tbl_width_chars=200): print(out.drop("R"))
    print(out.group_by("country").agg(pl.col("acc0", "acc1", "est").sum()))
