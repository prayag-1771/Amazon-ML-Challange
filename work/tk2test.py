import sys, time
sys.path.insert(0, ".")
import polars as pl
from src.features import load_side
from src.token_stats import token_table_ctx, token_features
s1, q = load_side("test")
t = time.time(); tab = token_table_ctx("test", s1, q, force=True); print("table", tab.shape, time.time()-t)
print(tab.filter((pl.col("country") == "France") & (pl.col("side") == "xq") & pl.col("tok").is_in(["groupe", "holding", "sas"])).sort("tok", "kind", "pos"))
c = pl.read_parquet("../work/pruned_test.parquet").head(200000)
t = time.time(); f = token_features(c, "test", s1, q); print("feat", time.time()-t)
print(f.select(pl.col("^x(q|1)_k.*$")).describe())
