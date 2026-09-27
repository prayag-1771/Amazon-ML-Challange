import sys; sys.path.insert(0,".")
import polars as pl
W="../work/"
def load(split, scores, lab):
    s1=pl.read_parquet(W+f"norm_{split}_s1.parquet",columns=["entity_id","country","name_core","addr_core"]).rename({"entity_id":"s1_id","name_core":"a","addr_core":"aa"})
    q=pl.concat([pl.read_parquet(W+f"norm_{split}_s{k}.parquet",columns=["entity_id","name_core","addr_core","business_name"]) for k in (2,3)]).rename({"entity_id":"s23_id","name_core":"b","addr_core":"qa"})
    d=pl.read_parquet(W+scores,columns=["s1_id","s23_id","p"]+(["label"] if lab else [])).join(s1,on="s1_id").join(q,on="s23_id")
    d=d.with_columns(pl.col("a").str.split(" ").list.unique().alias("ta"),pl.col("b").str.split(" ").list.unique().alias("tb"))
    d=d.with_columns(pl.col("tb").list.set_difference("ta").alias("add"),pl.col("ta").list.set_difference("tb").alias("rem"))
    d=d.filter((pl.col("add").list.len()==1)&(pl.col("rem").list.len()==0)&(pl.col("a").str.len_chars()>0)).with_columns(pl.col("add").list.first().alias("w"))
    # does the query's name_core exist as an S1 name in the same country? (sibling entity present)
    s1n=s1.select("country",pl.col("a").alias("b")).unique().with_columns(pl.lit(1).alias("q_in_s1"))
    d=d.join(s1n,on=["country","b"],how="left").with_columns(pl.col("q_in_s1").fill_null(0))
    # query's best p anywhere
    best=pl.read_parquet(W+scores,columns=["s23_id","p"]).group_by("s23_id").agg(pl.col("p").max().alias("qbest"))
    d=d.join(best,on="s23_id")
    # position of added token: last word of raw name?
    d=d.with_columns((pl.col("b").str.split(" ").list.last()==pl.col("w")).alias("at_end"))
    # S1 name freq of the token
    tf=s1.select("country",pl.col("a").str.split(" ")).explode("a").group_by("country","a").len().rename({"a":"w","len":"s1f"})
    return d.join(tf,on=["country","w"],how="left").fill_null(0)
v=load("train","valid_scores_stage2.parquet",True)
t=load("test","test_scores_ens_v8.parquet",False)
import polars as pl
pl.Config.set_tbl_rows(80)
g=v.group_by("country","w").agg(pl.len(),pl.col("label").mean().alias("y"),pl.col("q_in_s1").mean().alias("q_in_s1"),pl.col("at_end").mean().alias("end"),(pl.col("qbest")-pl.col("p")).mean().alias("better_else"),pl.col("s1f").first()).filter(pl.col("len")>=60).sort("len",descending=True)
print(g.head(40))
g=t.filter(pl.col("country")=="France").group_by("w").agg(pl.len(),(pl.col("p")>=.7).mean().alias("acc"),pl.col("q_in_s1").mean().alias("q_in_s1"),pl.col("at_end").mean().alias("end"),(pl.col("qbest")-pl.col("p")).mean().alias("better_else"),pl.col("s1f").first()).filter(pl.col("len")>=60).sort("len",descending=True)
print(g.head(40))
