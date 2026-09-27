import polars as pl
W="../work/"
q=pl.concat([pl.read_parquet(W+f"norm_test_s{k}.parquet",columns=["entity_id","country","addr_empty"]) for k in (2,3)]).rename({"entity_id":"s23_id"})
s=pl.read_parquet(W+"test_scores_stage2_v12.parquet")
top=s.sort("p2",descending=True).unique("s23_id",keep="first")
m=pl.read_csv("../output/matching_results.tsv",separator="\t",quote_char=None).with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids")!="").select(pl.col("matched_entity_ids").alias("s23_id"),pl.lit(1).alias("acc"))
x=q.join(top.select("s23_id","p2"),on="s23_id",how="left").join(m,on="s23_id",how="left").with_columns(pl.col("acc").fill_null(0))
print(x.group_by("country","addr_empty").agg(pl.len(),pl.col("p2").is_null().mean().round(3).alias("nocand"),pl.col("acc").mean().round(3),(pl.col("p2")>=.5).mean().round(3).alias("p>=.5"),(pl.col("p2")>=.2).mean().round(3).alias("p>=.2")).sort("country","addr_empty").rows())
