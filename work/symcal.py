import sys; sys.path.insert(0, ".")
import polars as pl
exec(open("../work/symcorr.py", encoding="utf-8").read().split("v, s1v = prep")[0])
t, _ = prep("test_scores_stage2_v10.parquet", "test")
x = pl.read_parquet(W + "test_top_symcorr.parquet").join(t.select("s23_id", *K, "up"), on="s23_id")
x = x.filter(pl.col("dg").is_not_null() & (pl.col("p2") > 0.02))
th = pl.col("country").replace_strict(T)
Rv = {"US": 0.93, "India": 1.0, "France": 0.95}
g = x.group_by(K).agg(pl.col("up").sum().alias("nu"), (~pl.col("up")).sum().alias("nd"),
    pl.col("p2").filter(pl.col("up")).sum().alias("sum_p2_up"), pl.col("p3").filter(pl.col("up")).sum().alias("sum_p3_up"),
    (pl.col("p2") >= th).filter(pl.col("up")).sum().alias("acc2_up"), (pl.col("p3") >= th).filter(pl.col("up")).sum().alias("acc3_up"),
    (pl.col("p2") >= th).filter(~pl.col("up")).sum().alias("acc_dn"), pl.col("p2").filter(~pl.col("up")).sum().alias("sum_p2_dn"))
g = g.with_columns((pl.col("nd") * pl.col("country").replace_strict(Rv)).round(0).alias("true_up_est"))
for r in g.filter(pl.col("nu") > 1000).sort("nu", descending=True).select(K + ["nu", "nd", "true_up_est", "sum_p2_up", "sum_p3_up", "acc2_up", "acc3_up", "acc_dn", "sum_p2_dn"]).with_columns(pl.col("sum_p2_up", "sum_p3_up", "sum_p2_dn").round(0)).rows(): print(r)
print(g.group_by("country").agg(pl.col("nu", "nd", "true_up_est", "sum_p2_up", "sum_p3_up", "acc2_up", "acc3_up").sum()).rows())
