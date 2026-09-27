import sys; sys.path.insert(0, ".")
import polars as pl
from src.io_utils import load_ground_truth
W = "../work/"
gt = load_ground_truth()
src = pl.concat([pl.read_parquet(W + f"norm_train_s{i}.parquet", columns=["entity_id", "addr_empty"]).with_columns(pl.lit(i).alias("src")) for i in (2, 3)]).rename({"entity_id": "s23_id"})
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
g = gt.join(src, on="s23_id").join(s1, on="s1_id")
print("queries per source", src.group_by("src").len().rows(), "S1", len(s1), "gold", len(gt), "gold q unique", gt["s23_id"].n_unique())
c = g.group_by("s1_id", "src").len()
full = s1.join(pl.DataFrame({"src": [2, 3]}), how="cross").join(c, on=["s1_id", "src"], how="left").with_columns(pl.col("len").fill_null(0))
print(full.group_by("country", "src", "len").len("n").sort("country", "src", "len").rows())
e = g.group_by("s1_id", "src").agg(pl.col("addr_empty").sum().alias("ne"), pl.len())
print("empty per s1/src", e.group_by("src", "ne").len("n").sort("src", "ne").rows())
