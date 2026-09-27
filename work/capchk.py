import polars as pl
W="../work/"
def rd(f):
    m=pl.read_csv(f,separator="\t",infer_schema=False)
    return m.select(pl.col(m.columns[0]).alias("s1_id"),pl.col(m.columns[1]).str.split(",").alias("s23_id")).explode("s23_id").filter(pl.col("s23_id").str.len_chars()>0)
q=pl.concat([pl.read_parquet(W+f"norm_test_s{i}.parquet",columns=["entity_id"]).with_columns(pl.lit(i).alias("src")) for i in (2,3)]).rename({"entity_id":"s23_id"})
c=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
m=rd("../output/matching_results.tsv").join(q,on="s23_id").join(c,on="s1_id")
k=m.group_by("s1_id","country").agg((pl.col("src")==2).sum().alias("n2"),(pl.col("src")==3).sum().alias("n3"))
print("n2>5",k.filter(pl.col("n2")>5).group_by("country").agg(pl.len(),(pl.col("n2")-5).sum()).rows())
print("n3>6",k.filter(pl.col("n3")>6).group_by("country").agg(pl.len(),(pl.col("n3")-6).sum()).rows())
print("n2 dist",k.group_by("country","n2").len().filter(pl.col("n2")>=4).sort("country","n2").rows())
print("n3 dist",k.group_by("country","n3").len().filter(pl.col("n3")>=5).sort("country","n3").rows())
