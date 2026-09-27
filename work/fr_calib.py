"""Label-free precision check per p band. Train (labelled US/India top-1 pairs): house-number agreement rate among
true matches (e1) and among non-matches (e0). On test assigned pairs, a band's agreement rate r implies
precision ~ (r - e0) / (e1 - e0). Compared per country with the model's own mean p."""
import polars as pl

W = "../work/"
cols = ["entity_id", "country", "nums"]


def numeq(d):
    return d.with_columns(
        pl.when((pl.col("m1") == "") | (pl.col("mq") == "")).then(None)
          .otherwise(pl.col("mq").str.split(" ").list.contains(pl.col("m1").str.split(" ").list.first())).alias("eq"))


def attach(split, pairs):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=cols)
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=cols) for k in (2, 3)])
    return numeq(pairs.join(s1.rename({"entity_id": "s1_id", "nums": "m1"}), on="s1_id")
                      .join(q.select(pl.col("entity_id").alias("s23_id"), pl.col("nums").alias("mq")), on="s23_id"))


gt = pl.read_parquet(W + "gt_pairs.parquet").with_columns(pl.lit(1).alias("y"))
tr = pl.read_parquet(W + "pruned_train.parquet", columns=["s1_id", "s23_id", "p0_qrank"]).filter(pl.col("p0_qrank") == 1)
tr = attach("train", tr.drop("p0_qrank")).join(gt, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("y").fill_null(0))
e = tr.drop_nulls("eq").group_by("country", "y").agg(pl.col("eq").mean()).sort("country", "y")
print("train top-1 agreement by label:\n", e)
e0 = e.filter(pl.col("y") == 0)["eq"].mean(); e1 = e.filter(pl.col("y") == 1)["eq"].mean()
print(f"e0={e0:.4f} e1={e1:.4f}")

sc = pl.read_parquet(W + "test_scores_lgb_v4a.parquet")
b = sc.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= 0.5)
b = attach("test", b)
b = b.with_columns(pl.col("p").cut([0.75, 0.9, 0.97, 0.99, 0.999], left_closed=True).alias("band"))
r = b.group_by("country", "band").agg(pl.len().alias("n"), pl.col("p").mean().alias("mean_p"),
                                      pl.col("eq").is_null().mean().alias("na"), pl.col("eq").mean().alias("agree"))
r = r.with_columns(((pl.col("agree") - e0) / (e1 - e0)).clip(0, 1).alias("impl_prec"))
pl.Config.set_tbl_rows(40)
print(r.sort("country", "band"))
