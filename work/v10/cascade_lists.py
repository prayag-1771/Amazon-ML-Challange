"""Cascade candidate lists: pairs with stage-1 p = mean(lgb_v5cf, lgb_v6k) >= PCUT, for valid and test."""
import sys
sys.path.insert(0, ".")
import polars as pl
W = "../work/"
for split in ("valid", "test"):
    a = pl.read_parquet(W + f"{split}_scores_lgb_v5cf.parquet", columns=["s1_id", "s23_id", "p"]).rename({"p": "pa"})
    b = pl.read_parquet(W + f"{split}_scores_lgb_v6k.parquet", columns=["s1_id", "s23_id", "p"]).rename({"p": "pb"})
    d = a.join(b, on=["s1_id", "s23_id"]).with_columns(((pl.col("pa") + pl.col("pb")) / 2).alias("p"))
    for c in (0.001, 0.02):
        x = d.filter(pl.col("p") >= c).select("s1_id", "s23_id")
        x.write_parquet(W + f"cascade_{split}_p{str(c)[2:]}.parquet")
        print(split, c, f"{d.height:,} -> {x.height:,}", flush=True)
