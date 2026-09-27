import sys; sys.path.insert(0, "."); sys.path.insert(0, "../work")
import polars as pl
from symfix import prep, K
from src.config import is_valid_expr
v, _ = prep("valid_scores_stage2.parquet", "train", is_valid_expr("s1_id"))
R = pl.read_parquet("../work/sym_R.parquet")
x = v.filter(pl.col("dg").is_not_null() & (pl.col("p2") > 0.02))
g = x.group_by(K).agg(pl.col("up").sum().alias("nu"), (~pl.col("up")).sum().alias("nd"),
    pl.col("label").filter(pl.col("up")).sum().alias("tu"), pl.col("label").filter(~pl.col("up")).sum().alias("td"),
    pl.col("p2").filter(pl.col("up")).sum().round(0).alias("sp")).join(R, on=["country", "dg"])
g = g.with_columns((pl.col("R") * pl.col("nd")).round(0).alias("est")).filter(pl.col("nd") >= 20).sort("nu", descending=True)
with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200): print(g)
