import sys; sys.path.insert(0,".")
import polars as pl
W="../work/"
def run(split, feats, scores, lab):
    s1=pl.read_parquet(W+f"norm_{split}_s1.parquet",columns=["entity_id","country","name_core"]).rename({"entity_id":"s1_id","name_core":"a"})
    q=pl.concat([pl.read_parquet(W+f"norm_{split}_s{k}.parquet",columns=["entity_id","name_core"]) for k in (2,3)]).rename({"entity_id":"s23_id","name_core":"b"})
    d=pl.read_parquet(W+scores).join(s1,on="s1_id").join(q,on="s23_id")
    if lab: d=d.join(pl.read_parquet(W+"valid_feats.parquet",columns=["s1_id","s23_id","label"]),on=["s1_id","s23_id"])
    d=d.filter(pl.col("p")>0.02)
    d=d.with_columns(pl.col("a").str.split(" ").list.unique().alias("ta"),pl.col("b").str.split(" ").list.unique().alias("tb"))
    d=d.with_columns(pl.col("ta").list.set_difference("tb").alias("da"),pl.col("tb").list.set_difference("ta").alias("db"),pl.col("ta").list.set_intersection("tb").list.len().alias("ni"))
    sw=d.filter((pl.col("da").list.len()==1)&(pl.col("db").list.len()==1)&(pl.col("ni")>=1))
    sw=sw.with_columns(pl.col("da").list.first().alias("x"),pl.col("db").list.first().alias("y"))
    # token frequency among S1 names of that country
    tf=s1.select("country",pl.col("a").str.split(" ")).explode("a").group_by("country","a").len().rename({"a":"tok","len":"f"})
    n=s1.group_by("country").len().rename({"len":"N"})
    sw=sw.join(tf.rename({"tok":"x","f":"fx"}),on=["country","x"],how="left").join(tf.rename({"tok":"y","f":"fy"}),on=["country","y"],how="left").join(n,on="country").fill_null(0)
    sw=sw.with_columns(((pl.min_horizontal("fx","fy")/pl.col("N"))>0.002).alias("bothfreq"),
        (pl.col("x").str.len_chars()>=3)&(pl.col("y").str.len_chars()>=3)&(~pl.col("x").str.contains(r"\d"))&(~pl.col("y").str.contains(r"\d")))
    agg=[pl.len(),pl.col("p").mean().alias("p"),(pl.col("p")>=0.7).mean().alias("acc")]
    if lab: agg.append(pl.col("label").mean().alias("y"))
    print(split, sw.group_by("country","bothfreq").agg(agg).sort("country","bothfreq"))
    if not lab:
        print(sw.filter(pl.col("bothfreq")&(pl.col("country")=="France")&(pl.col("p")>=0.7)).group_by("x","y").len().sort("len",descending=True).head(15))
    else:
        print(sw.filter(pl.col("bothfreq")).group_by("country",pl.col("p").cut([0.3,0.7,0.95,0.999])).agg(pl.len(),pl.col("label").mean()).sort("country","p"))
run("train",None,"valid_scores_stage2.parquet",True)
run("test",None,"test_scores_ens_v8.parquet",False)
