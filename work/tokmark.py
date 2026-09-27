"""Supervised extra-token match rates on train top-1 pairs (US/India) vs label-free proxy rates on test France."""
import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
W = "../work/"
pr = pl.read_parquet(W + "pruned_train.parquet", columns=["s1_id", "s23_id", "p0_qrank"]).filter(pl.col("p0_qrank") == 1)
gt = pl.read_parquet(W + "gt_pairs.parquet").with_columns(pl.lit(1).alias("y"))
pr = pr.join(gt, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("y").fill_null(0))
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "name_full", "country", "addr_core"]).rename({"entity_id": "s1_id", "name_full": "n1", "addr_core": "a1"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=["entity_id", "name_full", "addr_core"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "name_full": "nq", "addr_core": "aq"})
d = pr.join(s1, on="s1_id").join(q, on="s23_id").with_columns((pl.col("a1") == pl.col("aq")).alias("same_addr"))
d = d.with_columns(pl.col("nq").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).alias("xq"),
                   pl.col("n1").str.split(" ").list.set_difference(pl.col("nq").str.split(" ")).alias("x1"))
print("share same exact addr_core:", d.group_by("country").agg(pl.col("same_addr").mean()))
for side in ("xq", "x1"):
    e = d.select("country", "y", "same_addr", pl.col(side).alias("tok")).explode("tok").drop_nulls("tok")
    g = e.group_by("tok").agg(pl.len().alias("n"), pl.col("y").mean().alias("rate"),
                              pl.col("y").filter(pl.col("same_addr")).mean().alias("rate_same_addr"),
                              pl.col("same_addr").sum().alias("n_same")).filter(pl.col("n") >= 3000)
    print(side, "LOWEST (distractor markers)"); print(g.sort("rate").head(40))
    print(side, "HIGHEST"); print(g.sort("rate", descending=True).head(15))
