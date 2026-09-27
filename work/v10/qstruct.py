import sys; sys.path.insert(0,".")
import polars as pl, numpy as np
from src.io_utils import load_ground_truth
W="../work/"
gt=load_ground_truth()
for split in ["train","test"]:
    s1=pl.read_parquet(W+f"norm_{split}_s1.parquet",columns=["entity_id","country"])
    for k in (2,3):
        q=pl.read_parquet(W+f"norm_{split}_s{k}.parquet",columns=["entity_id","country","addr_empty"])
        a=q.group_by("country").agg(pl.len().alias("nq"),pl.col("addr_empty").mean().alias("empty")).join(s1.group_by("country").len().rename({"len":"ns1"}),on="country").with_columns((pl.col("nq")/pl.col("ns1")).alias("q_per_s1"))
        print(split,"s",k,a.sort("country").rows())
# row-order relation on train
s1=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id"]).with_row_index("r1")
for k in (2,3):
    q=pl.read_parquet(W+f"norm_train_s{k}.parquet",columns=["entity_id"]).with_row_index("rq")
    d=gt.join(s1.rename({"entity_id":"s1_id"}),on="s1_id").join(q.rename({"entity_id":"s23_id"}),on="s23_id")
    print("src",k,"spearman row order",np.corrcoef(d["r1"].rank().to_numpy(),d["rq"].rank().to_numpy())[0,1], "id sample", d.head(3).select("s1_id","s23_id").rows())
