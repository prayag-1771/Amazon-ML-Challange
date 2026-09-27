import sys; sys.path.insert(0, ".")
import polars as pl
W = "../work/"
g = pl.read_parquet(W + "fnb2.parquet").filter(pl.col("in_union").is_null())
s1 = pl.read_parquet(W + "norm_train_s1.parquet").rename({"entity_id": "s1_id"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{i}.parquet") for i in (2, 3)]).rename({"entity_id": "s23_id"})
print(s1.columns)
g = g.join(s1.select("s1_id", pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1")), on="s1_id").join(
    q.select("s23_id", pl.col("business_name").alias("nq"), pl.col("business_address").alias("aq")), on="s23_id")
for c, e in [("US", True), ("India", True), ("India", False), ("US", False)]:
    print(f"== {c} empty={e}")
    for r in g.filter((pl.col("country") == c) & (pl.col("q_empty") == e)).sample(14, seed=3).select("n1", "a1", "nq", "aq").rows():
        print("  ", r)
