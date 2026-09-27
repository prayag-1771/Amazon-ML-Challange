"""Per country: distribution of each query's best p (test v5) and how many S1 would change under other thresholds."""
import polars as pl

W = "../work/"
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"])
sc = pl.read_parquet(W + "test_scores_lgb_v4a.parquet")
b = sc.sort("p", descending=True).unique("s23_id", keep="first").join(s1, left_on="s1_id", right_on="entity_id")
edges = [0, 0.1, 0.3, 0.5, 0.65, 0.75, 0.85, 0.95, 0.99, 1.01]
b = b.with_columns(pl.col("p").cut(edges[1:-1], left_closed=True).alias("band"))
t = b.group_by("country", "band").agg(pl.len().alias("n")).with_columns(
    (pl.col("n") / pl.col("n").sum().over("country")).round(4).alias("share"))
pl.Config.set_tbl_rows(40)
print(t.pivot(on="country", index="band", values="share").sort("band"))
nq = b.group_by("country").agg(pl.len().alias("queries"), (pl.col("p") >= 0.75).mean().alias("assigned"))
ns1 = s1.group_by("country").agg(pl.len().alias("s1"))
print(nq.join(ns1, on="country").with_columns((pl.col("queries") / pl.col("s1")).alias("q_per_s1")))
