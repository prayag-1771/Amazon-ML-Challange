import sys; sys.path.insert(0, ".")
import polars as pl
from src.io_utils import load_ground_truth
from src.config import is_valid_expr
W="../work/"
gt=load_ground_truth(); print("gt", gt.columns, gt.height)
q=[pl.read_parquet(W+f"norm_train_s{i}.parquet",columns=["entity_id"]).height for i in (2,3)]; print("train queries", sum(q))
v=pl.read_parquet(W+"valid_scores_stage2.parquet")
vt=v.sort("p2",descending=True).unique("s23_id",keep="first")
c1=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
vt=vt.join(c1,on="s1_id")
th=pl.col("country").replace_strict({"US":0.75,"India":0.8,"France":0.75},default=0.75)
print("val queries w/ cand", vt.group_by("country").agg(pl.len(), (pl.col("p2")>=th).mean().alias("acc"), pl.col("label").mean().alias("top1_true")).rows())
s1v=c1.filter(is_valid_expr("s1_id"))
print("val S1", s1v.group_by("country").len().rows(), "true pairs per S1", gt.join(s1v, left_on=gt.columns[0], right_on="s1_id").group_by("country").len().rows())
t=pl.read_parquet(W+"test_scores_stage2_v12.parquet")
c=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
tt=t.sort("p2",descending=True).unique("s23_id",keep="first").join(c,on="s1_id")
nq=sum(pl.read_parquet(W+f"norm_test_s{i}.parquet",columns=["entity_id"]).height for i in (2,3))
print("test queries", nq, "with cand", tt.height)
print("test top1", tt.group_by("country").agg(pl.len(), (pl.col("p2")>=th).mean().alias("acc"), (pl.col("p2")<0.1).mean().alias("lt01")).sort("country").rows())
print("test S1", c.group_by("country").len().sort("country").rows())
