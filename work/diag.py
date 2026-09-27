import polars as pl
from src.io_utils import load_ground_truth
gt = load_ground_truth()
s1 = pl.read_parquet("../work/norm_train_s1.parquet").select(pl.col("entity_id").alias("s1_id"), pl.col("name_core").alias("n1"), pl.col("addr_core").alias("a1"), pl.col("state").alias("st1"), "country")
q = pl.concat([pl.read_parquet(f"../work/norm_train_s{i}.parquet") for i in (2,3)]).select(pl.col("entity_id").alias("s23_id"), pl.col("name_core").alias("n2"), pl.col("addr_core").alias("a2"), pl.col("state").alias("st2"))
p = gt.sample(200000, seed=0).join(s1, on="s1_id").join(q, on="s23_id")
print(p.group_by("country").agg((pl.col("n1")==pl.col("n2")).mean().alias("name_eq"), (pl.col("st1")==pl.col("st2")).mean().alias("state_eq"), pl.len()))
with open("../work/diag.txt","w",encoding="utf-8") as f:
    for r in p.filter((pl.col("n1")!=pl.col("n2"))&(pl.col("country")=="India")).head(40).iter_rows(named=True):
        f.write(f"{r['n1']!r:45} | {r['n2']!r:45} || {r['a1'][:50]!r} | {r['a2'][:50]!r}\n")
