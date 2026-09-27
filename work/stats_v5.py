import polars as pl
W = "../work/"
s1 = pl.read_parquet(W + "raw_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
out = {}
for m in ("lgb_v3", "lgb_v4a"):
    te = pl.read_parquet(W + f"test_scores_{m}.parquet").sort("p", descending=True).unique("s23_id", keep="first")
    t = 0.75
    pr = te.filter(pl.col("p") >= t).select("s1_id", "s23_id")
    out[m] = pr
    g = pr.join(s1, on="s1_id").group_by("country").len().join(s1.group_by("country").len("n"), on="country")
    print(m, len(pr), g.with_columns((pl.col("len") / pl.col("n")).round(3).alias("per_s1")).sort("country"))
a, b = out["lgb_v3"], out["lgb_v4a"]
print("kept from v4:", a.join(b, on=["s1_id", "s23_id"], how="semi").height / len(a))
d = pl.concat([a.join(b, on=["s1_id","s23_id"], how="anti").with_columns(pl.lit("dropped").alias("k")),
               b.join(a, on=["s1_id","s23_id"], how="anti").with_columns(pl.lit("added").alias("k"))]).join(s1, on="s1_id")
print(d.group_by("country", "k").len().sort("country", "k"))
