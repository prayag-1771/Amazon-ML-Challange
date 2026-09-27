import sys; sys.path.insert(0, ".")
import polars as pl, numpy as np
from src import india_addr as ia
W = "../work/"
casc = pl.read_parquet(W + "cascade_test_p001.parquet").select("s23_id")
d = ia.score(ia.pairs("test", casc)); d.write_parquet(W + "india_addr_test_pairs.parquet")
p = pl.read_parquet(W + "v10/ac2_test.parquet")
print("pairs mine", len(d), "peer", len(p), "q", d["q_row"].n_unique(), p["q_row"].n_unique())
j = d.join(p, on=["q_row", "s1_row"], suffix="_p")
print("joined", len(j))
for f in ia.F:
    x = (j[f].cast(pl.Float64) - j[f + "_p"].cast(pl.Float64)).abs()
    if x.max() > 1e-5: print(f, "diff rows", (x > 1e-5).sum(), "max", x.max())
