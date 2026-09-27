import sys; sys.path.insert(0,".")
import polars as pl
from src.config import is_valid_expr
W="../work/"
def prep(split, sc):
    s1=pl.read_parquet(W+f"norm_{split}_s1.parquet",columns=["entity_id","country","name_core","nums"]).rename({"entity_id":"s1_id","name_core":"n1","nums":"u1"})
    q=pl.concat([pl.read_parquet(W+f"norm_{split}_s{i}.parquet",columns=["entity_id","name_core","nums"]) for i in (2,3)]).rename({"entity_id":"s23_id","name_core":"nq","nums":"uq"})
    x=sc.sort("p2",descending=True).unique("s23_id",keep="first").join(s1,on="s1_id").join(q,on="s23_id")
    t1=pl.col("n1").str.split(" "); tq=pl.col("nq").str.split(" ")
    a=pl.col("u1").str.split(" ").list.first().cast(pl.Int64,strict=False); b=pl.col("uq").str.split(" ").list.first().cast(pl.Int64,strict=False)
    x=x.with_columns(tq.list.set_difference(t1).alias("A"),t1.list.set_difference(tq).list.len().alias("nd"),(b==a).alias("eq"),((b-a).abs().is_between(1,25)).alias("sh"))
    x=x.filter(pl.col("A").list.len()==1).with_columns(pl.col("A").list.first().alias("at"))
    return x, s1
def stats(split, sc, lab):
    x,s1=prep(split,sc)
    if split=="train": s1=s1.filter(is_valid_expr("s1_id")); x=x.join(s1.select("s1_id"),on="s1_id")
    ns=s1.group_by("country").len("ns")
    fq=s1.select("country",pl.col("n1").str.split(" ").list.unique().alias("t")).explode("t").group_by("country","t").len("f").rename({"t":"at"})
    g=x.group_by("country","at").agg((pl.col("eq")).sum().alias("e"),(pl.col("sh")&(pl.col("nd")==0)).sum().alias("sa"),
        *( [pl.col("label").filter(pl.col("eq")).mean().alias("prec_eq")] if lab else [(pl.col("p2")>=0.75).filter(pl.col("eq")).mean().alias("acc_eq")]))
    g=g.join(ns,on="country").join(fq,on=["country","at"],how="left").with_columns(
        (pl.col("e")/pl.col("ns")*1000).round(2).alias("eqK"),(pl.col("sa")/pl.col("ns")*1000).round(2).alias("saK"),(pl.col("f").fill_null(0)/pl.col("ns")*1000).round(2).alias("fK"))
    return g.with_columns((pl.col("eqK")/(pl.col("fK")+1)).round(3).alias("eq_f"))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20)
v=stats("train",pl.read_parquet(W+"valid_scores_stage2.parquet"),True)
t=stats("test",pl.read_parquet(W+"test_scores_stage2_v11.parquet"),False)
for c,d in (("US",v),("India",v),("France",t)):
    print(c); print(d.filter(pl.col("country")==c).sort("e",descending=True).head(25).drop("country","ns","f"))
v.write_parquet(W+"vt_val.parquet"); t.write_parquet(W+"vt_test.parquet")
