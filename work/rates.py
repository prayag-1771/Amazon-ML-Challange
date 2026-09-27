import polars as pl
W="../work/"
sc = pl.read_parquet(W+"test_scores_lgb_v6k.parquet")
s1 = pl.read_parquet(W+"norm_test_s1.parquet", columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
q = pl.concat([pl.read_parquet(W+f"norm_test_s{k}.parquet", columns=["entity_id","country","addr_empty"]) for k in (2,3)]).rename({"entity_id":"s23_id"})
best = sc.sort("p", descending=True).unique("s23_id", keep="first")
q = q.join(best, on="s23_id", how="left")
print(q.group_by("country").agg(pl.len(), (pl.col("p")>=0.7).mean().alias("assigned"), pl.col("p").is_null().mean().alias("nocand"),
   ((pl.col("p")<0.7)&(pl.col("p")>=0.1)).mean().alias("mid"), pl.col("addr_empty").mean().alias("aempty")))
print(q.group_by("country","addr_empty").agg(pl.len(), (pl.col("p")>=0.7).mean().alias("assigned")).sort("country","addr_empty"))
a = q.filter(pl.col("p")>=0.7).group_by("s1_id").len()
x = s1.join(a, on="s1_id", how="left").with_columns(pl.col("len").fill_null(0))
print(x.group_by("country").agg(pl.len().alias("n"), pl.col("len").mean().alias("per_s1"), (pl.col("len")==0).mean().alias("empty")))
print(x.group_by("country", pl.col("len").clip(0,8).alias("k")).agg(pl.len().alias("c")).sort("country","k").pivot(on="country", index="k", values="c"))
# train truth
gt = pl.read_parquet(W+"gt_pairs.parquet")
t1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
tq = pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet", columns=["entity_id","country","addr_empty"]) for k in (2,3)]).rename({"entity_id":"s23_id"})
tq = tq.join(gt, on="s23_id", how="left")
print(tq.group_by("country").agg(pl.len(), pl.col("s1_id").is_not_null().mean().alias("matched")))
print(tq.group_by("country","addr_empty").agg(pl.len(), pl.col("s1_id").is_not_null().mean().alias("matched")).sort("country","addr_empty"))
y = t1.join(gt.group_by("s1_id").len(), on="s1_id", how="left").with_columns(pl.col("len").fill_null(0))
print(y.group_by("country").agg(pl.len().alias("n"), pl.col("len").mean().alias("per_s1"), (pl.col("len")==0).mean().alias("empty")))
print(y.group_by("country", pl.col("len").clip(0,8).alias("k")).agg(pl.len().alias("c")).sort("country","k").pivot(on="country", index="k", values="c"))
