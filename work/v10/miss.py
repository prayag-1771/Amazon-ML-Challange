import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W = "../work/"
exec(open("../work/v10/gap.py").read().split("K = [")[0].split("VAR = ")[0])
exec("VAR = " + open("../work/v10/gap.py").read().split("VAR = ")[1].split("K = [")[0])
v = prep("train", pl.read_parquet(W + "valid_scores_stage2.parquet"))
s1c = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
v = v.filter(is_valid_expr("s1_id"))
T = {"US": .75, "India": .80}
b = [0, .02, .1, .3, .5, .75, .8, .9, 1.01]
for k, e in [("name=", "missing"), ("name=", "far"), ("name=", "near-"), ("swap1_typo", "eq"), ("other", "eq")]:
    x = v.filter((pl.col("k") == k) & (pl.col("e") == e)).with_columns(pl.col("p2").cut(b[1:-1]).alias("bin"))
    print(f"\n## {k} {e} (val)")
    print(x.group_by("country", "bin").agg(pl.len(), pl.col("label").mean().round(3)).sort("country", "bin").pivot(on="country", index="bin", values=["len", "label"]))
f = pl.read_parquet(W + "v10/fr_top1.parquet").with_columns(pl.col("p2").cut(b[1:-1]).alias("bin"))
for k, e in [("name=", "missing"), ("name=", "far"), ("name=", "near-"), ("swap1_typo", "eq"), ("other", "eq")]:
    print(f"\n## FR {k} {e}")
    print(f.filter((pl.col("k") == k) & (pl.col("e") == e)).group_by("bin").agg(pl.len(), pl.col("acc").mean().round(3)).sort("bin"))
