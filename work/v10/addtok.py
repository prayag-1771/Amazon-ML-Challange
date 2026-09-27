import sys; sys.path.insert(0,".")
import polars as pl
W="../work/"
def diffs(split, scores, lab):
    s1=pl.read_parquet(W+f"norm_{split}_s1.parquet",columns=["entity_id","country","name_core"]).rename({"entity_id":"s1_id","name_core":"a"})
    q=pl.concat([pl.read_parquet(W+f"norm_{split}_s{k}.parquet",columns=["entity_id","name_core"]) for k in (2,3)]).rename({"entity_id":"s23_id","name_core":"b"})
    d=pl.read_parquet(W+scores,columns=["s1_id","s23_id","p"]+(["label"] if lab else [])).join(s1,on="s1_id").join(q,on="s23_id").filter(pl.col("p")>0.02)
    d=d.with_columns(pl.col("a").str.split(" ").list.unique().alias("ta"),pl.col("b").str.split(" ").list.unique().alias("tb"))
    d=d.with_columns(pl.col("tb").list.set_difference("ta").alias("add"),pl.col("ta").list.set_difference("tb").alias("rem"))
    return d
v=diffs("train","valid_scores_stage2.parquet",True)
t=diffs("test","test_scores_ens_v8.parquet",False)
def top_add(d,country,lab):
    x=d.filter(pl.col("country")==country).explode("add").filter(pl.col("add").is_not_null())
    agg=[pl.len(),(pl.col("p")>=0.7).mean().alias("acc")]
    if lab: agg.append(pl.col("label").mean().alias("y"))
    return x.group_by("add").agg(agg).sort("len",descending=True).head(45)
for c in ["US","India"]: print(c,"ADDED tokens (val)");print(top_add(v,c,True))
print("France ADDED tokens (test)");print(top_add(t,"France",False))
for c in ["US","India"]:
    x=t.filter(pl.col("country")==c).explode("add").filter(pl.col("add").is_not_null())
    print(c,"test",x.group_by("add").agg(pl.len(),(pl.col("p")>=0.7).mean().alias("acc")).sort("len",descending=True).head(0))
