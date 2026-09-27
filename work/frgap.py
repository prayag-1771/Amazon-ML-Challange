import sys; sys.path.insert(0, ".")
import polars as pl
W="../work/"
def rd(f):
    m=pl.read_csv(f,separator="\t",infer_schema=False)
    return m.select(pl.col(m.columns[0]).alias("s1_id"),pl.col(m.columns[1]).str.split(",").alias("s23_id")).explode("s23_id").filter(pl.col("s23_id").str.len_chars()>0)
for split in ("train","test"):
    q=pl.concat([pl.read_parquet(W+f"norm_{split}_s{i}.parquet",columns=["entity_id","country","addr_empty"]) for i in (2,3)]).rename({"entity_id":"s23_id"})
    s1=pl.read_parquet(W+f"norm_{split}_s1.parquet",columns=["entity_id","country"])
    ns=s1.group_by("country").len("ns")
    if split=="test":
        c=pl.read_parquet(W+"cascade_test_p001.parquet",columns=["s23_id"]).unique().with_columns(pl.lit(True).alias("hasc"))
        a=rd("../output/matching_results.tsv").select("s23_id").with_columns(pl.lit(True).alias("acc"))
        x=q.join(c,on="s23_id",how="left").join(a,on="s23_id",how="left").fill_null(False)
        print(x.group_by("country").agg(pl.len().alias("nq"),pl.col("hasc").sum(),pl.col("acc").sum(),(pl.col("addr_empty")==1).sum().alias("qempty"),pl.col("acc").filter(pl.col("addr_empty")==1).sum().alias("acc_empty"),pl.col("hasc").filter(pl.col("addr_empty")==1).sum().alias("hasc_empty")).join(ns,on="country").with_columns((pl.col("nq")/pl.col("ns")).alias("q/S1"),(pl.col("acc")/pl.col("ns")).alias("acc/S1"),(pl.col("acc_empty")/pl.col("ns")).alias("accE/S1"),(pl.col("qempty")/pl.col("ns")).alias("qE/S1")).sort("country"))
    else:
        from src.io_utils import load_ground_truth
        gt=load_ground_truth().select("s23_id").with_columns(pl.lit(True).alias("true"))
        x=q.join(gt,on="s23_id",how="left").fill_null(False)
        print(x.group_by("country").agg(pl.len().alias("nq"),pl.col("true").sum(),(pl.col("addr_empty")==1).sum().alias("qempty"),pl.col("true").filter(pl.col("addr_empty")==1).sum().alias("true_empty")).join(ns,on="country").with_columns((pl.col("nq")/pl.col("ns")).alias("q/S1"),(pl.col("true")/pl.col("ns")).alias("true/S1"),(pl.col("true_empty")/pl.col("ns")).alias("trueE/S1"),(pl.col("qempty")/pl.col("ns")).alias("qE/S1")).sort("country"))
