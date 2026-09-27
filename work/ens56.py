import sys
sys.path.insert(0, ".")
import polars as pl
from src.metric import macro_f05
from src.config import is_valid_expr
W = "../work/"
gt = pl.read_parquet(W + "gt_pairs.parquet")
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
a = pl.read_parquet(W + "valid_scores_lgb_v5cf.parquet").select("s1_id", "s23_id", pl.col("p").alias("pa"))
b = pl.read_parquet(W + "valid_scores_lgb_v6k.parquet").select("s1_id", "s23_id", pl.col("p").alias("pb"))
d = a.join(b, on=["s1_id", "s23_id"]).with_columns(((pl.col("pa") + pl.col("pb")) / 2).alias("p"))
top = d.sort("p", descending=True).unique("s23_id", keep="first").join(s1v, on="s1_id", how="semi")
print([(t, round(macro_f05(s1v["s1_id"], top.filter(pl.col("p") >= t), gt, by=s1v)["f05"], 5)) for t in (0.65, 0.7, 0.75, 0.8)])
