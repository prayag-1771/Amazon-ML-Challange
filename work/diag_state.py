import polars as pl
from src.io_utils import load_ground_truth
gt = load_ground_truth()
s1 = pl.read_parquet("../work/norm_train_s1.parquet").select(pl.col("entity_id").alias("s1_id"), pl.col("state").alias("st1"), "country")
q = pl.concat([pl.read_parquet(f"../work/norm_train_s{i}.parquet") for i in (2,3)]).select(pl.col("entity_id").alias("s23_id"), pl.col("state").alias("st2"), pl.col("business_address").alias("a2"))
p = gt.join(s1, on="s1_id").join(q, on="s23_id")
print(p.group_by("country").agg(pl.len(), (pl.col("st2")=="").mean().alias("q_empty"), (pl.col("st1")=="").mean().alias("s1_empty"),
      ((pl.col("st1")!=pl.col("st2")) & (pl.col("st2")!="") & (pl.col("st1")!="")).mean().alias("conflict")))
bad = p.filter((pl.col("st1")!=pl.col("st2")) & (pl.col("st2")!="") & (pl.col("st1")!=""))
print(bad.group_by("country","st1","st2").len().sort("len",descending=True).head(25))
with open("../work/state_bad.txt","w",encoding="utf-8") as f:
    for r in bad.sample(40, seed=0).iter_rows(named=True): f.write(f"{r['st1']} | {r['st2']} | {r['a2']}\n")
t = pl.concat([pl.read_parquet(f"../work/norm_test_s{i}.parquet") for i in (1,2,3)])
print(t.group_by("country").agg((pl.col("state")=="").mean().alias("empty"), pl.col("state").n_unique().alias("n_states")))
