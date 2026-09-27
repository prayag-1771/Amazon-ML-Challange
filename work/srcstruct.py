import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
W="../work/"
def src(split):
    return pl.concat([pl.read_parquet(W+f"norm_{split}_s{k}.parquet", columns=["entity_id"]).with_columns(pl.lit(k).alias("src")) for k in (2,3)]).rename({"entity_id":"s23_id"})
def struct(pairs, s1, tag):
    g = pairs.group_by("s1_id").agg((pl.col("src")==2).sum().alias("n2"), (pl.col("src")==3).sum().alias("n3"))
    x = s1.join(g, on="s1_id", how="left").with_columns(pl.col("n2","n3").fill_null(0))
    return x.group_by("country").agg(pl.len(), ((pl.col("n2")==0)&(pl.col("n3")>0)).mean().alias("only3"), ((pl.col("n3")==0)&(pl.col("n2")>0)).mean().alias("only2"),
        (pl.col("n2")==0).mean().alias("n2_0"), (pl.col("n3")==0).mean().alias("n3_0"), pl.col("n2").mean().alias("m2"), pl.col("n3").mean().alias("m3"),
        (pl.col("n2")>=4).mean().alias("n2_4p"), (pl.col("n3")>=5).mean().alias("n3_5p")).with_columns(pl.lit(tag).alias("tag")).sort("country")
gt = pl.read_parquet(W+"gt_pairs.parquet").join(src("train"), on="s23_id")
s1t = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
print(struct(gt, s1t, "train_truth"))
v = pl.read_parquet(W+"valid_scores_lgb_v6k.parquet").sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p")>=0.7).join(src("train"), on="s23_id")
print(struct(v, s1t.filter(is_valid_expr("s1_id")), "val_pred"))
te = pl.read_parquet(W+"test_scores_lgb_v6k.parquet").sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p")>=0.7).join(src("test"), on="s23_id")
s1e = pl.read_parquet(W+"norm_test_s1.parquet", columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
print(struct(te, s1e, "test_pred"))
