import sys; sys.path.insert(0, ".")
import polars as pl
import src.stage2 as S
from src.io_utils import load_ground_truth
from src.config import is_valid_expr
from src.metric import macro_f05
W="../work/"
gt=load_ground_truth()
s1v=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
a=pl.read_parquet(W+"valid_scores_stage2_v11.parquet").select("s1_id","s23_id",pl.col("p2").alias("pa"))
b=pl.read_parquet(W+"valid_scores_stage2.parquet")
d=b.join(a,on=["s1_id","s23_id"],how="left")
print("null v11", d["pa"].null_count())
for w in (0,0.25,0.5,0.75,1.0):
    x=d.with_columns((w*pl.col("pa")+(1-w)*pl.col("p2")).alias("q"))
    top=x.sort("q",descending=True).unique("s23_id",keep="first").join(s1v,on="s1_id")
    for t in (0.7,0.75):
        th=pl.col("country").replace_strict({"US":t,"India":t+0.05},default=t)
        mm=macro_f05(s1v["s1_id"],top.filter(pl.col("q")>=th).select("s1_id","s23_id"),gt,by=s1v)
        print(f"w_v11={w} t={t}: {mm['f05']:.5f} (US {mm['f05_US']:.5f} IN {mm['f05_India']:.5f})",flush=True)
