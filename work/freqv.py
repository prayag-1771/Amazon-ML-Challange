import sys
sys.path.insert(0, ".")
import polars as pl
from src.io_utils import load_source

W = "../work/"
split, lo, hi, n, rels, c = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split(","), sys.argv[6]
tr = "train" if split == "val" else "test"

top = pl.read_parquet(W + f"numrel_{split}.parquet").filter(
    (pl.col("country") == c) & pl.col("rel").is_in(rels) & (pl.col("p") > lo) & (pl.col("p") <= hi)
)
s1 = load_source(tr, 1).select(
    pl.col("entity_id").alias("s1_id"),
    pl.col("business_name").alias("name1"),
    pl.col("business_address").alias("a1")
)
q = pl.concat([load_source(tr, s) for s in (2, 3)]).select(
    pl.col("entity_id").alias("s23_id"),
    pl.col("business_name").alias("nameq"),
    pl.col("business_address").alias("aq")
)

m = top.sample(min(n, top.height), seed=3).join(s1, on="s1_id").join(q, on="s23_id")
print("Total matching count:", top.height)
for r in m.iter_rows(named=True):
    lbl = f"y={r.get('label')}" if 'label' in r else ""
    print(f"p={r['p']:.4f} {r['rel']} {lbl}\n   S1: {r['name1']} | {r['a1']}\n   Q : {r['nameq']} | {r['aq']}")
