import polars as pl
W = "../work/"
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country", "nums"]).rename({"entity_id": "s1_id", "nums": "n1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{k}.parquet", columns=["entity_id", "nums"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "nums": "nq"})
def asg(name, t):
    d = pl.read_parquet(W + f"test_scores_{name}.parquet")
    return d.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= t).select("s1_id", "s23_id")
a = asg("lgb_v5cf", 0.75); b = asg("lgb_v6k", 0.70)
eq = (pl.when((pl.col("n1") == "") | (pl.col("nq") == "")).then(None)
      .otherwise(pl.col("nq").str.split(" ").list.contains(pl.col("n1").str.split(" ").list.first()).cast(pl.Float32)))
for lab, u, v in (("only_v6k", b, a), ("only_v5cf", a, b), ("common", a.join(b, on=["s1_id", "s23_id"]), None)):
    d = u if v is None else u.join(v, on=["s1_id", "s23_id"], how="anti")
    d = d.join(s1, on="s1_id").join(q, on="s23_id").with_columns(eq.alias("numeq"))
    print(lab, d.group_by("country").agg(pl.len(), pl.col("numeq").mean().round(3), pl.col("numeq").null_count().alias("nonum")).sort("country"))
