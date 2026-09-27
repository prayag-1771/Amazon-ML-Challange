import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W = "../work/"
for sp in ("train", "test"):
    s1 = pl.read_parquet(W + f"norm_{sp}_s1.parquet", columns=["entity_id", "country"])
    if sp == "train": s1 = s1.filter(is_valid_expr("entity_id"))
    print(sp, s1.group_by("country").len().sort("country").to_dicts())
v = pl.read_parquet(W + "valid_scores_stage2.parquet")
t = pl.read_parquet(W + "test_scores_stage2_v10.parquet")
c1 = pl.concat([pl.read_parquet(W + f"norm_{sp}_s1.parquet", columns=["entity_id", "country"]) for sp in ("train", "test")]).rename({"entity_id": "s1_id"})
for nm, d in (("val", v), ("test", t)):
    top = d.sort("p2", descending=True).unique("s23_id", keep="first").join(c1, on="s1_id")
    g = top.with_columns(pl.col("p2").cut([0.05, 0.2, 0.5, 0.75, 0.9, 0.98], labels=["<.05", ".05-.2", ".2-.5", ".5-.75", ".75-.9", ".9-.98", ">.98"]).alias("b"))
    n1 = c1.join(d.select("s1_id").unique(), on="s1_id").group_by("country").len().rename({"len": "ns1"})
    x = g.group_by("country", "b").len().join(n1, on="country").with_columns((pl.col("len") / pl.col("ns1")).round(4).alias("per_s1"))
    print(nm); print(x.pivot(on="b", index="country", values="per_s1").sort("country"))
