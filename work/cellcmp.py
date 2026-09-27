import sys; sys.path.insert(0, "."); sys.path.insert(0, "../work")
import polars as pl
from symfix import prep, K, T
t, _ = prep("test_scores_stage2_v10.parquet", "test")
v9 = pl.read_parquet("../work/test_scores_stage2_v9.parquet").select("s1_id", "s23_id", pl.col("p2").alias("p9"))
t = t.join(v9, on=["s1_id", "s23_id"], how="left")
th = pl.col("country").replace_strict(T)
x = t.filter(pl.col("dg").is_not_null() & pl.col("up"))
g = x.group_by(K).agg(pl.len().alias("nu"), (pl.col("p") >= th).sum().alias("acc_p"), (pl.col("p9") >= th).sum().alias("acc_v9"), (pl.col("p2") >= th).sum().alias("acc_v10"))
dn = t.filter(pl.col("dg").is_not_null() & ~pl.col("up")).group_by(K).agg(pl.len().alias("nd"))
g = g.join(dn, on=K, how="left").sort("acc_v10", descending=True)
with pl.Config(tbl_rows=25, tbl_width_chars=200): print(g.head(25))
print(g.group_by("country").agg(pl.col("acc_p", "acc_v9", "acc_v10").sum()))
print("totals", t.group_by("country").agg((pl.col("p") >= th).sum(), (pl.col("p9") >= th).sum(), (pl.col("p2") >= th).sum()))
