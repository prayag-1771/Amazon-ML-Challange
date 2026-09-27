import polars as pl
W="../work/"
def rd(f):
    m=pl.read_csv(f,separator="\t",infer_schema=False)
    return m.select(pl.col(m.columns[0]).alias("s1_id"),pl.col(m.columns[1]).str.split(",").alias("s23_id")).explode("s23_id").filter(pl.col("s23_id").str.len_chars()>0)
s1=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country","name_core","nums"]).rename({"entity_id":"s1_id"})
q=pl.concat([pl.read_parquet(W+f"norm_test_s{i}.parquet",columns=["entity_id","name_core","nums"]) for i in (2,3)]).rename({"entity_id":"s23_id","name_core":"nq","nums":"uq"})
m12=rd("../output/matching_results.tsv"); vf=rd(W+"probe11_fr_vfix/matching_results.tsv")
s12=pl.read_parquet(W+"test_scores_stage2_v12.parquet"); s11=pl.read_parquet(W+"test_scores_stage2_v11.parquet")
t12=s12.sort("p2",descending=True).unique("s23_id",keep="first")
for nm,x in (("vfix-only",vf.join(m12,on=["s1_id","s23_id"],how="anti")),("v12-only",m12.join(vf,on=["s1_id","s23_id"],how="anti"))):
    x=x.join(s1,on="s1_id").filter(pl.col("country")=="France").join(q,on="s23_id")
    x=x.join(s11.select("s1_id","s23_id",pl.col("p2").alias("p2_11")),on=["s1_id","s23_id"],how="left").join(s12.select("s1_id","s23_id",pl.col("p2").alias("p2_12")),on=["s1_id","s23_id"],how="left")
    x=x.join(t12.select("s23_id",pl.col("s1_id").alias("top12")),on="s23_id",how="left")
    x=x.with_columns((pl.col("name_core")==pl.col("nq")).alias("nsame"),(pl.col("nums").str.split(" ").list.first()==pl.col("uq").str.split(" ").list.first()).alias("neq"),(pl.col("top12")==pl.col("s1_id")).alias("istop"))
    print(nm,x.height,x.group_by("nsame","neq","istop").len().sort("len",descending=True).rows())
    print("  p2_11 mean",x["p2_11"].mean(),"p2_12 mean",x["p2_12"].mean(), "p2_12 null", x["p2_12"].null_count())
    for r in x.sample(12,seed=3).select("p2_11","p2_12","name_core","nums","nq","uq").rows(): print("  ",r)
