import polars as pl
def rd(p,c):
    return pl.read_csv(p,separator="\t",quote_char=None,schema_overrides={c:pl.Utf8}).with_columns(pl.col(c).fill_null("").str.split(",").list.eval(pl.element().filter(pl.element()!="")).list.len().alias("n")).rename({"source1_entity_id":"s1_id"}).select("s1_id","n")
m=rd("output/matching_results.tsv","matched_entity_ids").join(pl.read_parquet("work/norm_test_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"}),on="s1_id")
g=pl.read_parquet("work/gt_pairs.parquet").group_by("s1_id").len("n")
t=pl.read_parquet("work/norm_train_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"}).join(g,on="s1_id",how="left").with_columns(pl.col("n").fill_null(0))
for nm,x in (("TRAIN truth",t),("TEST pred",m)):
    print(nm)
    print(x.group_by("country").agg(pl.col("n").mean().round(3).alias("mean"),*[(pl.col("n")==k).mean().round(4).alias(f"n{k}") for k in range(0,8)],(pl.col("n")>=8).mean().round(4).alias("n8+")).sort("country"))
