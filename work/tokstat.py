import sys; sys.path.insert(0,".")
import polars as pl
from src.config import is_valid_expr
W="../work/"
def tokstats(split):
    """Per (country, token) rates per 1000 S1 over each query's stage-1 top-1 (label-free)."""
    s1=pl.read_parquet(W+f"norm_{split}_s1.parquet",columns=["entity_id","country","name_core","nums"]).rename({"entity_id":"s1_id","name_core":"n1","nums":"u1"})
    q=pl.concat([pl.read_parquet(W+f"norm_{split}_s{i}.parquet",columns=["entity_id","name_core","nums"]) for i in (2,3)]).rename({"entity_id":"s23_id","name_core":"nq","nums":"uq"})
    sc=pl.read_parquet(W+("valid_scores_stage2.parquet" if split=="train" else "test_scores_stage2_v11.parquet"),columns=["s1_id","s23_id","p"])
    if split=="train": s1=s1.filter(is_valid_expr("s1_id"))
    x=sc.sort("p",descending=True).unique("s23_id",keep="first").join(s1,on="s1_id").join(q,on="s23_id")
    t1=pl.col("n1").str.split(" "); tq=pl.col("nq").str.split(" ")
    a=pl.col("u1").str.split(" ").list.first().cast(pl.Int64,strict=False); b=pl.col("uq").str.split(" ").list.first().cast(pl.Int64,strict=False)
    x=x.with_columns(tq.list.set_difference(t1).alias("A"),t1.list.set_difference(tq).alias("D"),(b==a).fill_null(False).alias("eq"),((b-a).abs().is_between(1,25)).fill_null(False).alias("sh"))
    x=x.filter(pl.col("A").list.len()==1).with_columns(pl.col("A").list.first().alias("at"),pl.col("D").list.first().alias("dt"),pl.col("D").list.len().alias("nd"))
    ns=s1.group_by("country").len("ns")
    sw=x.filter((pl.col("nd")==1)&pl.col("eq"))
    ad=x.filter(pl.col("nd")==0)
    tin=sw.group_by("country","at").len("sw_in").rename({"at":"tok"})
    tout=sw.group_by("country","dt").len("sw_out").rename({"dt":"tok"})
    aeq=ad.filter(pl.col("eq")).group_by("country","at").len("add_eq").rename({"at":"tok"})
    ash=ad.filter(pl.col("sh")).group_by("country","at").len("add_sh").rename({"at":"tok"})
    fq=s1.select("country",pl.col("n1").str.split(" ").list.unique().alias("tok")).explode("tok").group_by("country","tok").len("s1f")
    st=fq
    for z in (tin,tout,aeq,ash): st=st.join(z,on=["country","tok"],how="full",coalesce=True)
    st=st.join(ns,on="country").with_columns([(pl.col(c).fill_null(0)/pl.col("ns")*1000).cast(pl.Float32).alias(c) for c in ("s1f","sw_in","sw_out","add_eq","add_sh")]).drop("ns")
    return st.filter(pl.col("tok").is_not_null() & (pl.col("tok")!=""))
if __name__=="__main__":
    pl.Config.set_tbl_rows(40)
    for sp in ("train","test"):
        st=tokstats(sp); st.write_parquet(W+f"tokstat_{sp}.parquet")
        for c in st["country"].unique().sort():
            print(sp,c); print(st.filter(pl.col("country")==c).sort("sw_in",descending=True).head(14).with_columns((pl.col("sw_in")/(pl.col("sw_out")+0.2)).round(1).alias("io")).drop("country"))
