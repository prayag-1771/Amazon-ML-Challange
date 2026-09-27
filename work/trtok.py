"""Train truth: single-added-token (query = S1 name_core + 1 token) label rate per token and country, top-1 prune candidate."""
import polars as pl
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(250)
W = "../work/"
gt = pl.read_parquet(W + "gt_pairs.parquet").with_columns(pl.lit(1).alias("label"))
pr = pl.read_parquet(W + "pruned_train.parquet", columns=["s1_id", "s23_id", "p0"]).sort("p0", descending=True).unique("s23_id", keep="first")
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "name_core", "country"]).rename({"entity_id": "s1_id", "name_core": "n1"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{i}.parquet", columns=["entity_id", "name_core"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq"})
d = pr.join(s1, on="s1_id").join(q, on="s23_id").join(gt, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("label").fill_null(0))
d = d.with_columns(pl.col("nq").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).alias("add"),
                   pl.col("n1").str.split(" ").list.set_difference(pl.col("nq").str.split(" ")).alias("drop"))
one = d.filter((pl.col("add").list.len() == 1) & (pl.col("drop").list.len() == 0)).with_columns(pl.col("add").list.first().alias("tok"))
a = one.group_by("country", "tok").agg(pl.len(), pl.col("label").mean().round(3).alias("lab"), pl.col("p0").mean().round(3).alias("p0"))
for c in ("US", "India"):
    print(c); print(a.filter(pl.col("country") == c).sort("len", descending=True).head(50))
# tokens that are distractor (lab<0.3) with n>=200 and noise (lab>0.7)
one.write_parquet(W + "trtok_one.parquet")
