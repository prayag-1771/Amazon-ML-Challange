import sys; sys.path.insert(0, ".")
import polars as pl
from src.metric import macro_f05
from src.io_utils import load_ground_truth
from src.config import is_valid_expr
import src.stage2 as s2
s1v = pl.read_parquet("../work/norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
v = pl.read_parquet("../work/valid_scores_stage2.parquet").join(s1v, on="s1_id").sort("p2", descending=True).unique("s23_id", keep="first")
m = macro_f05(s1v["s1_id"], v.filter(pl.col("p2") >= pl.col("country").replace_strict(s2.T)).select("s1_id", "s23_id"), load_ground_truth(), by=s1v)
print({k: round(x, 5) for k, x in m.items() if isinstance(x, float)})
