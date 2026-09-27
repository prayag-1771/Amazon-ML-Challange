import polars as pl
W = "../work/"
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country", "nums"]).rename({"entity_id": "s1_id", "nums": "n1"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=["entity_id", "nums"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "nums": "nq"})
eq = (pl.when((pl.col("n1") == "") | (pl.col("nq") == "")).then(None)
      .otherwise(pl.col("nq").str.split(" ").list.contains(pl.col("n1").str.split(" ").list.first()).cast(pl.Float32)))
def top(name, t):
    d = pl.read_parquet(W + f"valid_scores_{name}.parquet")
    return d.sort("p", descending=True).unique("s23_id", keep="first")
a = top("lgb_v5cf", 0); b = top("lgb_v6k", 0)
aa = a.filter(pl.col("p") >= 0.75).select("s1_id", "s23_id", "label"); bb = b.filter(pl.col("p") >= 0.70).select("s1_id", "s23_id", "label")
for lab, u, v in (("only_v6k", bb, aa), ("only_v5cf", aa, bb)):
    d = u.join(v, on=["s1_id", "s23_id"], how="anti").join(s1, on="s1_id").join(q, on="s23_id").with_columns(eq.alias("numeq"))
    print(lab, d.group_by("country").agg(pl.len(), pl.col("label").mean().round(3), pl.col("numeq").mean().round(3)).sort("country"))
# numeq by label in uncertain band
d = b.filter(pl.col("p").is_between(0.3, 0.98)).join(s1, on="s1_id").join(q, on="s23_id").with_columns(eq.alias("numeq"))
print(d.group_by("country", "label").agg(pl.len(), pl.col("numeq").mean().round(3)).sort("country", "label"))
