import sys
sys.path.insert(0, ".")
import polars as pl
from src.io_utils import load_source
W = "../work/"
lo, hi, n, rels = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(",")
c = sys.argv[5] if len(sys.argv) > 5 else "France"
top = pl.read_parquet(W+"numrel_test.parquet").filter((pl.col("country")==c)&pl.col("rel").is_in(rels)&(pl.col("p")>lo)&(pl.col("p")<=hi))
s1 = load_source("test",1).select(pl.col("entity_id").alias("s1_id"),pl.col("business_name").alias("name1"),pl.col("business_address").alias("a1"))
q = pl.concat([load_source("test",s) for s in (2,3)]).select(pl.col("entity_id").alias("s23_id"),pl.col("business_name").alias("nameq"),pl.col("business_address").alias("aq"))
m = top.sample(min(n, top.height), seed=2).join(s1,on="s1_id").join(q,on="s23_id")
print(top.height)
for r in m.iter_rows(named=True):
    print(f"p={r['p']:.3f} {r['rel']}\n   S1: {r['name1']} | {r['a1']}\n   Q : {r['nameq']} | {r['aq']}")
