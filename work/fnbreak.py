import sys; sys.path.insert(0,".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W="../work/"; T={"US":0.75,"India":0.80}
s1v=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1v,on="s1_id")
d=pl.read_parquet(W+"valid_scores_stage2.parquet")
top=d.sort("p2",descending=True).unique("s23_id",keep="first").select("s23_id",pl.col("s1_id").alias("top_s1"),pl.col("p2").alias("top_p2"))
g=gt.join(d.select("s1_id","s23_id","p2"),on=["s1_id","s23_id"],how="left").join(top,on="s23_id",how="left")
th=pl.col("country").replace_strict(T)
g=g.with_columns(pl.when(pl.col("p2").is_null()).then(pl.lit("no_cand")).when(pl.col("top_s1")!=pl.col("s1_id")).then(pl.lit("lost_argmax"))
   .when(pl.col("p2")<th).then(pl.lit("below_th")).otherwise(pl.lit("TP")).alias("k"))
print(g.group_by("country","k").len().sort("country","k"))
# FP
acc=top.join(d.select(pl.col("s1_id").alias("top_s1"),"s23_id","label"),on=["top_s1","s23_id"]).join(s1v.rename({"s1_id":"top_s1"}),on="top_s1").filter(pl.col("top_p2")>=th)
print("FP",acc.group_by("country").agg((pl.col("label")==0).sum(),pl.len()))
# per-S1 impact: F0.5 loss contributions
per=g.group_by("s1_id","country").agg(pl.len().alias("n"),(pl.col("k")=="TP").sum().alias("tp"),*(((pl.col("k")==c).sum()).alias(c) for c in ["no_cand","lost_argmax","below_th"]))
fp=acc.filter(pl.col("label")==0).group_by(pl.col("top_s1").alias("s1_id")).len("fp")
per=s1v.join(per.drop("country"),on="s1_id",how="left").join(fp,on="s1_id",how="left").fill_null(0)
b=0.25
per=per.with_columns(pl.when(pl.col("n")+pl.col("fp")==0).then(1.0).otherwise((1+b)*pl.col("tp")/((1+b)*pl.col("tp")+b*(pl.col("n")-pl.col("tp"))+pl.col("fp"))).alias("f"))
print(per.group_by("country").agg(pl.col("f").mean(),pl.len()))
# loss attribution: recompute f with each error type removed
for c in ["no_cand","lost_argmax","below_th"]:
    tp2=pl.col("tp")+pl.col(c)
    f2=pl.when(pl.col("n")+pl.col("fp")==0).then(1.0).otherwise((1+b)*tp2/((1+b)*tp2+b*(pl.col("n")-tp2)+pl.col("fp")))
    print(c, per.group_by("country").agg((f2-pl.col("f")).mean().round(5)).rows())
f2=pl.when(pl.col("n")==0).then(1.0).otherwise((1+b)*pl.col("tp")/((1+b)*pl.col("tp")+b*(pl.col("n")-pl.col("tp"))))
print("fp", per.group_by("country").agg((f2-pl.col("f")).mean().round(5)).rows())
print("singletons with FP:", per.filter((pl.col("n")==0)&(pl.col("fp")>0)).group_by("country").len().rows())
g.write_parquet(W+"fnbreak.parquet")
