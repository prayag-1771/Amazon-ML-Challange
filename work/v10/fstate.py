import polars as pl
W="../work/"
c=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country","state"]).rename({"entity_id":"s1_id","state":"s1_state"})
q=pl.concat([pl.read_parquet(W+f"norm_test_s{k}.parquet",columns=["entity_id","state","business_address"]) for k in (2,3)]).rename({"entity_id":"s23_id","state":"q_state"})
t=pl.read_parquet(W+"test_feats.parquet",columns=["s1_id","s23_id","state_eq","nc_eq","num_first_eq","addr_tok_jacc","city_jacc"]).join(c,on="s1_id").join(pl.read_parquet(W+"test_scores_ens_v8.parquet"),on=["s1_id","s23_id"])
print(t.group_by("country","state_eq").agg(pl.len(),(pl.col("p")>0.7).sum().alias("acc"),((pl.col("nc_eq")==1)&(pl.col("num_first_eq")==1)&(pl.col("addr_tok_jacc")>0.8)).sum().alias("strong"),pl.col("p").filter((pl.col("nc_eq")==1)&(pl.col("num_first_eq")==1)&(pl.col("addr_tok_jacc")>0.8)).mean().alias("p_strong")).sort("country","state_eq"))
fr=t.filter((pl.col("country")=="France")&(pl.col("state_eq")==0)).join(q,on="s23_id")
print(fr.group_by("s1_state","q_state").len().sort("len",descending=True).head(30))
