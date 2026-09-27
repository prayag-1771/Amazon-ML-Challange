import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
W = "../work/"
cols = ["entity_id", "name_core", "country"]
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=cols).rename({"entity_id": "s1_id", "name_core": "n1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{k}.parquet", columns=["entity_id", "name_core"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq"})
sc = pl.read_parquet(W + "test_scores_lgb_v5cf.parquet").sort("p", descending=True).unique("s23_id", keep="first")
d = sc.join(s1, on="s1_id").join(q, on="s23_id").with_columns(pl.col("n1").str.split(" ").alias("t1"), pl.col("nq").str.split(" ").alias("tq"))
d = d.with_columns(pl.col("tq").list.set_difference("t1").alias("extra"), pl.col("t1").list.set_difference("tq").alias("missing"),
                   pl.when(pl.col("p") > 0.999).then(pl.lit("conf")).when((pl.col("p") > 0.1) & (pl.col("p") < 0.75)).then(pl.lit("unc")).otherwise(None).alias("band")).filter(pl.col("band").is_not_null())
for c in ("France", "US"):
    x = d.filter(pl.col("country") == c)
    for b in ("conf", "unc"):
        y = x.filter(pl.col("band") == b)
        n = len(y)
        e = y.select(pl.col("extra").explode()).group_by("extra").len().sort("len", descending=True).head(15).with_columns((pl.col("len") / n).round(4).alias("rate"))
        m = y.select(pl.col("missing").explode()).group_by("missing").len().sort("len", descending=True).head(15).with_columns((pl.col("len") / n).round(4).alias("rate"))
        print(c, b, n, "no-extra share", round((y["extra"].list.len() == 0).mean(), 3), "no-missing share", round((y["missing"].list.len() == 0).mean(), 3))
        print(pl.concat([e.rename({"extra": "extra_tok", "len": "n_e", "rate": "r_e"}), m.rename({"missing": "missing_tok", "len": "n_m", "rate": "r_m"})], how="horizontal"))
# S1 name vocabulary: top tokens per country
v = s1.with_columns(pl.col("n1").str.split(" ")).explode("n1").group_by("country", "n1").len().sort("len", descending=True)
print(v.filter(pl.col("country") == "France").head(40)["n1"].to_list())
