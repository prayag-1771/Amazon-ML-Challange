import polars as pl

W = "../work/"
gt = pl.read_parquet(W + "gt_pairs.parquet")
def src(split):
    return pl.concat([pl.read_parquet(W + f"raw_{split}_s{i}.parquet", columns=["entity_id"]).with_columns(pl.lit(i).alias("src")) for i in (2, 3)]).rename({"entity_id": "s23_id"})
s1tr = pl.read_parquet(W + "raw_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
s1te = pl.read_parquet(W + "raw_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
def dist(pairs, s1, s):
    g = pairs.join(s, on="s23_id").group_by("s1_id", "src").len("k")
    g = s1.join(g, on="s1_id", how="left").with_columns(pl.col("k").fill_null(0))
    return g.group_by("country", "src", "k").len().with_columns((pl.col("len") / pl.col("len").sum().over("country", "src")).round(4).alias("frac"))
pl.Config.set_tbl_rows(200)
a = dist(gt, s1tr, src("train")).filter(pl.col("src").is_not_null())
te = pl.read_parquet(W + "test_scores_lgb_v3.parquet").sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= 0.75)
b = dist(te, s1te, src("test")).filter(pl.col("src").is_not_null())
print(a.join(b.rename({"len": "len_te", "frac": "frac_te"}).drop("country"), on=["src", "k"], how="full").filter(pl.col("country") == "US").sort("src", "k"))
print(b.pivot(on="country", index=["src", "k"], values="frac").sort("src", "k"))
