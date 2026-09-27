import polars as pl, sys
from src.normalize import normalize_source
b = sys.argv[1]; n = int(sys.argv[2])
fn = pl.read_parquet("../work/fn_v1.parquet").filter(pl.col("bucket") == b).sample(n, seed=int(sys.argv[3]))
s1 = normalize_source("train", 1).select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
q = pl.concat([normalize_source("train", s).select(pl.col("entity_id").alias("s23_id"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2")) for s in (2, 3)])
fn = fn.join(s1, on="s1_id").join(q, on="s23_id")
for r in fn.iter_rows(named=True):
    print(f"[{r['country']} top_p={r['top_p']}]\n  S1: {r['n1']} | {r['a1']}\n  Q : {r['n2']} | {r['a2']}")
