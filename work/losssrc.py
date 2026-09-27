import sys; sys.path.insert(0, ".")
import polars as pl
from src.io_utils import load_ground_truth
from src.config import is_valid_expr
W="../work/"
gt=load_ground_truth()
s1=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"}).filter(is_valid_expr("s1_id"))
g=gt.join(s1,on="s1_id")
q=pl.concat([pl.read_parquet(W+f"norm_train_s{i}.parquet",columns=["entity_id","addr_core"]) for i in (2,3)]).rename({"entity_id":"s23_id"})
g=g.join(q,on="s23_id").with_columns((pl.col("addr_core").fill_null("")=="").alias("qe"))
print(pl.read_parquet(W+"cand_train.parquet").columns)
c=pl.read_parquet(W+"cand_train.parquet")
if "s1_id" not in c.columns: print(c.head(3))
