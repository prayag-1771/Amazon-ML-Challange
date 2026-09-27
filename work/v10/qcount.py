import sys
sys.path.insert(0, ".")
import polars as pl
from src.io_utils import load_ground_truth
W = "../work/"
for split in ("train", "test"):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country"])
    for i in (2, 3):
        q = pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "country"])
        a = s1.group_by("country").len("n1").join(q.group_by("country").len("nq"), on="country", how="full", coalesce=True)
        print(split, f"S{i}", a.with_columns((pl.col("nq") / pl.col("n1")).round(4).alias("q_per_s1")).sort("country").rows())
gt = load_ground_truth()
print(gt.head(3))
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
g = gt.join(s1, on="s1_id").with_columns(pl.col(gt.columns[1]).str.slice(0, 2).alias("src"))
print(g.group_by("country", "src").len().join(s1.group_by("country").len("n1"), on="country").with_columns((pl.col("len") / pl.col("n1")).round(4)).sort("country", "src").rows())
for i in (2, 3):
    q = pl.read_parquet(W + f"norm_train_s{i}.parquet", columns=["entity_id", "country"])
    m = q.join(gt.rename({gt.columns[1]: "entity_id"}), on="entity_id", how="left")
    print(f"S{i} unmatched share by country", m.group_by("country").agg(pl.col("s1_id").is_null().mean().round(4)).rows())
