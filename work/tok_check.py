import sys, time
sys.path.insert(0, ".")
import polars as pl
from src.features import load_side
from src.token_stats import token_table, token_features
pl.Config.set_tbl_rows(40)
for split in ("train", "test"):
    t = time.time()
    s1, q = load_side(split)
    tab = token_table(split, s1, q, force=True)
    print(split, len(tab), f"{time.time()-t:.0f}s")
    for c in tab["country"].unique().sort():
        print(tab.filter((pl.col("country") == c) & (pl.col("side") == "xq")).sort("lfreq", descending=True).head(12))
    c = pl.read_parquet(f"../work/pruned_{split}.parquet", columns=["q_row", "s1_row"]).head(200000)
    t = time.time()
    f = token_features(c, split, s1, q)
    print(f"feat {time.time()-t:.1f}s"); print(f.describe())
