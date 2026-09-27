"""Validation F0.5 split by S1 'sibling' status (another S1 of the same country shares its exact address)."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05
pl.Config.set_tbl_width_chars(250)
W="../work/"
def sib(split):
    s1 = pl.read_parquet(W+f"norm_{split}_s1.parquet", columns=["entity_id","country","addr_core","name_core"]).rename({"entity_id":"s1_id"})
    s1 = s1.with_columns(pl.when(pl.col("addr_core").str.len_chars()>5).then(pl.len().over("country","addr_core")).otherwise(1).alias("naddr"),
                         pl.len().over("country","name_core").alias("nname"))
    return s1.with_columns(pl.col("naddr").clip(1,3).alias("sib"), pl.col("nname").clip(1,10).cut([1,3,9]).alias("nb"))
s1 = sib("train")
print("train share", s1.group_by("country","sib").len().with_columns((pl.col("len")/pl.col("len").sum().over("country")).alias("share")).sort("country","sib"))
print("test share", sib("test").group_by("country","sib").len().with_columns((pl.col("len")/pl.col("len").sum().over("country")).alias("share")).sort("country","sib"))
print("test name-dup share", sib("test").group_by("country","nb").len().with_columns((pl.col("len")/pl.col("len").sum().over("country")).alias("share")).sort("country","nb"))
print("train name-dup share", s1.group_by("country","nb").len().with_columns((pl.col("len")/pl.col("len").sum().over("country")).alias("share")).sort("country","nb"))
gt = pl.read_parquet(W+"gt_pairs.parquet")
v = s1.filter(is_valid_expr("s1_id"))
top = pl.read_parquet(W+"valid_scores_lgb_v6k.parquet").sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p")>=0.7)
for key in ("sib","nb"):
    by = v.select("s1_id", pl.concat_str([pl.col("country"), pl.col(key).cast(pl.String)], separator="_").alias("g"))
    m = macro_f05(v["s1_id"], top, gt, by=by)
    print(key, {k: round(x, 4) for k, x in m.items() if k.startswith("f05")})
