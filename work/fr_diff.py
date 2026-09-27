import polars as pl
W = "../work/"
r1 = pl.read_parquet(W + "raw_test_s1.parquet").select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"), "country")
rq = pl.concat([pl.read_parquet(W + f"raw_test_s{i}.parquet") for i in (2, 3)]).select(pl.col("entity_id").alias("s23_id"), pl.col("business_name").alias("nq"), pl.col("business_address").alias("aq"))
a = pl.read_parquet(W + "test_scores_lgb_v3.parquet").sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= 0.75)
b = pl.read_parquet(W + "test_scores_lgb_v4a.parquet").rename({"p": "p_new"})
d = a.join(b.select("s23_id", "s1_id", "p_new"), on=["s1_id", "s23_id"]).filter(pl.col("p_new") < 0.75).join(r1, on="s1_id").join(rq, on="s23_id")
d = d.filter(pl.col("country") == "France")
for r in d.sample(40, seed=5).iter_rows(named=True):
    print(f"{r['p']:.2f}->{r['p_new']:.2f} | {r['n1']} | {r['a1']}\n            | {r['nq']} | {r['aq']}")
