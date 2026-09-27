"""Label-free precision per p2 band from shift symmetry: among top-1 pairs whose first house number is shifted,
true ones split up/down with ratio R, distractors go up only. est_true = down * (1 + R). Checked against labels on val."""
import sys; sys.path.insert(0, ".")
import polars as pl
import src.stage2 as S
from src.config import is_valid_expr
W="../work/"
R=pl.read_parquet(W+"shift_R.parquet"); Rp=pl.read_parquet(W+"shift_Rp.parquet")
# all-candidate ratio (not only accepted): recompute R on val over all top-1 labelled pairs
B=[0,0.05,0.2,0.4,0.5,0.6,0.7,0.75,0.8,0.9,0.97,1.01]
def prep(split, sc, cmap):
    top=sc.sort("p2",descending=True).unique("s23_id",keep="first").join(cmap,on="s1_id")
    x=S.shift_cells(top,split).filter(pl.col("dg").is_not_null())
    return x.with_columns(pl.col("p2").cut(B[1:-1]).alias("band"))
cv=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"}).filter(is_valid_expr("s1_id"))
v=prep("train",pl.read_parquet(W+"valid_scores_stage2.parquet"),cv)
Rall=v.filter(pl.col("label")==1).group_by("dg").agg((pl.col("up").sum()/(~pl.col("up")).sum()).alias("R"))
print("R (all true, pooled)", Rall.sort("dg").rows())
def est(x):
    x=x.join(Rall,on="dg")
    return x.group_by("country","band").agg(pl.len().alias("n"),(~pl.col("up")).sum().alias("dn"),
        ((~pl.col("up")).cast(pl.Float64)*(1+pl.col("R"))).sum().alias("est_true"),
        *( [pl.col("label").sum().alias("true")] if "label" in x.columns else [])).with_columns((pl.col("est_true")/pl.col("n")).round(3).alias("est_prec")).sort("country","band")
e=est(v).with_columns((pl.col("true")/pl.col("n")).round(3).alias("prec"))
print(e.select("country","band","n","dn","est_prec","prec").rows())
ct=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
t=prep("test",pl.read_parquet(W+"test_scores_stage2_v12.parquet"),ct)
for r in est(t).select("country","band","n","dn","est_prec").rows(): print(r)
