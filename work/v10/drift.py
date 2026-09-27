import polars as pl
W="../work/"
c=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
cols=["s1_id","s23_id","nc_eq","num_first_eq","addr_tok_jacc","ac_tset","q_addr_empty","city_jacc","state_eq"]
v=pl.read_parquet(W+"valid_feats.parquet",columns=cols+["label"])
s1v=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
v=v.join(s1v,on="s1_id").join(pl.read_parquet(W+"valid_scores_stage2.parquet",columns=["s1_id","s23_id","p"]),on=["s1_id","s23_id"])
t=pl.read_parquet(W+"test_feats.parquet",columns=cols).join(c,on="s1_id").join(pl.read_parquet(W+"test_scores_ens_v8.parquet"),on=["s1_id","s23_id"])
def b(d): return d.with_columns(pl.col("addr_tok_jacc").cut([0.2,0.4,0.6,0.8]).alias("aj"))
print("== same core name + same first number, by address-token jaccard: pair share, label rate (val), mean p ==")
fv=b(v.filter((pl.col("nc_eq")==1)&(pl.col("num_first_eq")==1)))
print(fv.group_by("country","aj").agg(pl.len(),pl.col("label").mean().alias("y"),pl.col("p").mean().alias("p"),(pl.col("p")>=0.7).mean().alias("acc")).sort("country","aj"))
ft=b(t.filter((pl.col("nc_eq")==1)&(pl.col("num_first_eq")==1)))
print(ft.group_by("country","aj").agg(pl.len(),pl.col("p").mean().alias("p"),(pl.col("p")>=0.7).mean().alias("acc")).sort("country","aj"))
print("== addr_tok_jacc / ac_tset among confident (p>0.999) pairs: how addresses of sure matches look ==")
print(v.filter(pl.col("p")>0.999).group_by("country").agg(pl.col("addr_tok_jacc").mean(),pl.col("ac_tset").mean(),pl.col("city_jacc").mean(),(pl.col("state_eq")==1).mean().alias("st1"),(pl.col("state_eq")==-1).mean().alias("stm")))
print(t.filter(pl.col("p")>0.999).group_by("country").agg(pl.col("addr_tok_jacc").mean(),pl.col("ac_tset").mean(),pl.col("city_jacc").mean(),(pl.col("state_eq")==1).mean().alias("st1"),(pl.col("state_eq")==-1).mean().alias("stm")))
