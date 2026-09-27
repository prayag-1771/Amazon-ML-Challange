import sys; sys.path.insert(0,".")
import polars as pl
from rapidfuzz.distance import Levenshtein as L
from src.config import is_valid_expr
W="../work/"; T={"US":0.75,"India":0.80,"France":0.75}
def top1(split, sc):
    s1=pl.read_parquet(W+f"norm_{split}_s1.parquet",columns=["entity_id","country","name_core","nums"]).rename({"entity_id":"s1_id","name_core":"n1","nums":"u1"})
    q=pl.concat([pl.read_parquet(W+f"norm_{split}_s{i}.parquet",columns=["entity_id","name_core","nums"]) for i in (2,3)]).rename({"entity_id":"s23_id","name_core":"nq","nums":"uq"})
    if split=="train": s1=s1.filter(is_valid_expr("s1_id"))
    x=sc.sort("p2",descending=True).unique("s23_id",keep="first").join(s1,on="s1_id").join(q,on="s23_id")
    t1=pl.col("n1").str.split(" "); tq=pl.col("nq").str.split(" ")
    a=pl.col("u1").str.split(" ").list.first().cast(pl.Int64,strict=False); b=pl.col("uq").str.split(" ").list.first().cast(pl.Int64,strict=False)
    x=x.with_columns(tq.list.set_difference(t1).alias("A"),t1.list.set_difference(tq).alias("D"),(b==a).fill_null(False).alias("eq"))
    x=x.with_columns(pl.col("A").list.len().alias("na"),pl.col("D").list.len().alias("nd"))
    x=x.with_columns(pl.when((pl.col("na")==1)&(pl.col("nd")==1)).then(pl.lit("swap1")).when((pl.col("na")==1)&(pl.col("nd")==0)).then(pl.lit("add1")).otherwise(pl.lit("other")).alias("nm"),
                     pl.col("A").list.first().alias("at"),pl.col("D").list.first().alias("dt"))
    ns=s1.group_by("country").len("ns")
    fq=s1.select("country",pl.col("n1").str.split(" ").list.unique().alias("at")).explode("at").group_by("country","at").len("f")
    sw=x.filter(pl.col("nm")=="swap1").group_by("country","at").agg(pl.col("eq").sum().alias("se"),(~pl.col("eq")).sum().alias("sn"))
    sw=sw.join(ns,on="country").with_columns((pl.col("se")/pl.col("ns")*1000).alias("seK"),(pl.col("se")/(pl.col("sn")+1)).alias("sr"))
    x=x.join(sw.select("country","at","seK","sr"),on=["country","at"],how="left").join(fq,on=["country","at"],how="left").join(ns,on="country")
    x=x.with_columns(pl.col("seK","sr","f").fill_null(0),(pl.col("f").fill_null(0)/pl.col("ns")*1000).alias("fK"),
                     pl.col("p2")>=pl.col("country").replace_strict(T,default=0.75)).rename({"p2":"p2v"}) if False else x
    x=x.with_columns(pl.col("seK","sr","f").fill_null(0),(pl.col("f").fill_null(0)/pl.col("ns")*1000).alias("fK"),
                     (pl.col("p2")>=pl.col("country").replace_strict(T,default=0.75)).alias("acc"))
    sim=pl.struct("at","dt").map_elements(lambda r: L.normalized_similarity(r["at"] or "",r["dt"] or ""),return_dtype=pl.Float64)
    x=x.with_columns(sim.alias("sim"))
    x=x.with_columns(((pl.col("seK")>=5)&(pl.col("sr")>=1.5)).alias("var"),
        ((pl.col("fK")>=1)&(pl.col("sim")<0.5)&(pl.col("at").str.len_chars()>=4)&~pl.col("at").str.contains(r"\d")).alias("rw"))
    return x
pl.Config.set_tbl_rows(80)
v=top1("train",pl.read_parquet(W+"valid_scores_stage2.parquet"))
t=top1("test",pl.read_parquet(W+"test_scores_stage2_v11.parquet"))
print("VAR vocab"); print(v.filter(pl.col("var")).select("country","at").unique().sort("country","at").group_by("country").agg("at").rows())
print(t.filter(pl.col("var")).select("country","at").unique().sort("country","at").group_by("country").agg("at").rows())
def cat(x):
    return pl.when(pl.col("nm").is_in(["swap1","add1"])&pl.col("eq")&pl.col("var")).then(pl.concat_str(pl.col("nm"),pl.lit("_var"))) \
      .when((pl.col("nm")=="swap1")&pl.col("eq")&~pl.col("var")&pl.col("rw")).then(pl.lit("swap1_rw")).otherwise(pl.lit("-"))
v=v.with_columns(cat(v).alias("c")); t=t.with_columns(cat(t).alias("c"))
print(v.group_by("country","c","acc").agg(pl.len(),pl.col("label").mean().round(3).alias("prec"),(pl.len()/pl.col("ns").first()*1000).round(2).alias("perK")).sort("country","c","acc"))
print(t.group_by("country","c","acc").agg(pl.len(),(pl.len()/pl.col("ns").first()*1000).round(2).alias("perK")).sort("country","c","acc"))
v.select("s1_id","s23_id","country","c","acc","label","p2","at","dt").write_parquet(W+"vrule_val.parquet")
t.select("s1_id","s23_id","country","c","acc","p2","at","dt","n1","nq","u1","uq").write_parquet(W+"vrule_test.parquet")
