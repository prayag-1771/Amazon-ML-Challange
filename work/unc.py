"""Size of an 'uncertain' pair subset (label-free rule on cross-fitted ce_logit / p0) and share of v6k validation errors it covers."""
import sys; sys.path.insert(0, ".")
import polars as pl
W = "../work/"
v = pl.read_parquet(W+"valid_feats.parquet", columns=["q_row","s1_row","ce_logit","ce_gap_best","ce_gap_2nd","ce_qrank","label"])
s = pl.read_parquet(W+"valid_scores_lgb_v6k.parquet")
print(v.height, s.height, s.columns)
v = pl.concat([v, s.select("p")], how="horizontal")
v = v.with_columns(pl.col("p").rank("ordinal", descending=True).over("q_row").alias("r"), pl.col("p").max().over("q_row").alias("pmax"))
v = v.with_columns(((pl.col("r")==1)&(pl.col("p")>=0.7)).alias("pred"))
v = v.with_columns(((pl.col("pred")==1)&(pl.col("label")==0)).alias("fp"), ((pl.col("pred")==0)&(pl.col("label")==1)).alias("fn"))
print("errors", v["fp"].sum(), v["fn"].sum())
qmax = pl.col("ce_logit").max().over("q_row")
for lo, hi, g in ((-3, 5, 2), (-4, 6, 3), (-5, 7, 3), (-6, 8, 4)):
    sel = ((pl.col("ce_logit")>lo)&(pl.col("ce_logit")<hi)) | ((pl.col("ce_qrank")<=2)&(pl.col("ce_gap_2nd").abs()<g)&(qmax>lo))
    x = v.with_columns(sel.alias("sel"))
    print(lo, hi, g, "pairs", round(x["sel"].mean(),4), "fp cov", round(x.filter("fp")["sel"].mean(),3), "fn cov", round(x.filter("fn")["sel"].mean(),3))
# by p only (for reference)
for a, b in ((0.01,0.99),(0.003,0.997)):
    x = v.with_columns(((pl.col("p")>a)&(pl.col("p")<b)).alias("sel"))
    print("p", a, b, round(x["sel"].mean(),4), round(x.filter("fp")["sel"].mean(),3), round(x.filter("fn")["sel"].mean(),3))
print(v.filter("fn").select(pl.col("p").cut([0.001,0.01,0.1,0.3,0.7]).value_counts()).unnest("p"))
print(v.filter("fn").select(pl.col("ce_logit").cut([-8,-5,-3,0,3]).value_counts()).unnest("ce_logit"))
