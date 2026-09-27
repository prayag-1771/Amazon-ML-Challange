"""Acceptance / recall by missing-value cell.
train (labels): true match rate per cell;  valid: v15 recall + precision per cell;  test: v15 acceptance rate per cell."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "business_entity_resolution"))
import polars as pl
from src.config import is_valid_expr

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(230)
W = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/"
NULLTOK = r"(?i)(^|,)\s*(null|none|n/?a|nan|<null>|-)\s*(,|$)"


def queries(split):
    out = []
    for k in (2, 3):
        raw = pl.read_parquet(W + f"raw_{split}_s{k}.parquet", columns=["entity_id", "country", "business_address"])
        nrm = pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["state", "nums", "addr_empty", "alpha_comps"])
        d = pl.concat([raw, nrm], how="horizontal").with_columns(pl.lit(k).alias("src"))
        out.append(d.select(
            pl.col("entity_id").alias("s23_id"), "country", "src",
            pl.when(pl.col("addr_empty") == 1).then(pl.lit("0 empty"))
            .when(pl.col("business_address").str.contains(NULLTOK)).then(pl.lit("1 null-token"))
            .when(pl.col("state") == "").then(pl.lit("2 no-state"))
            .when(pl.col("nums") == "").then(pl.lit("3 no-number"))
            .otherwise(pl.lit("4 full")).alias("cell")))
    return pl.concat(out)


def read_tsv_pairs(path, col):
    t = pl.read_csv(path, separator="\t", quote_char=None, schema_overrides={col: pl.Utf8}).with_columns(pl.col(col).fill_null(""))
    return t.with_columns(pl.col(col).str.split(",")).explode(col).filter(pl.col(col) != "").select(
        pl.col("source1_entity_id").alias("s1_id"), pl.col(col).alias("s23_id"))


gt = pl.read_parquet(W + "gt_pairs.parquet")
# ---- train: true match rate by cell
qt = queries("train").join(gt.select("s23_id").with_columns(pl.lit(1).alias("y")), on="s23_id", how="left").with_columns(pl.col("y").fill_null(0))
tr = qt.group_by("country", "cell").agg(pl.len().alias("n_train"), pl.col("y").mean().round(4).alias("true_rate"))
# ---- valid: recall / precision by cell (queries of true pairs of validation S1; predictions on validation S1)
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id"]).rename({"entity_id": "s1_id"}).filter(is_valid_expr("s1_id"))
gtv = gt.join(s1v, on="s1_id", how="semi")
pv = pl.read_parquet(W + "pred15_v.parquet").select("s1_id", "s23_id").join(s1v, on="s1_id", how="semi")
qc = qt.select("s23_id", "country", "cell")
rec = gtv.join(qc, on="s23_id").join(pv.with_columns(pl.lit(1).alias("hit")), on=["s1_id", "s23_id"], how="left").group_by("country", "cell").agg(
    pl.col("hit").fill_null(0).mean().round(4).alias("val_recall"))
prc = pv.join(qc, on="s23_id").join(gtv.with_columns(pl.lit(1).alias("ok")), on=["s1_id", "s23_id"], how="left").group_by("country", "cell").agg(
    pl.col("ok").fill_null(0).mean().round(4).alias("val_prec"))
# ---- test: acceptance rate by cell under v15
qs = queries("test")
m = read_tsv_pairs(W + "sub15_indiaaddr/matching_results.tsv", "matched_entity_ids")
te = qs.join(m.select("s23_id").unique().with_columns(pl.lit(1).alias("a")), on="s23_id", how="left").group_by("country", "cell").agg(
    pl.len().alias("n_test"), pl.col("a").fill_null(0).mean().round(4).alias("test_accept"))
out = te.join(tr, on=["country", "cell"], how="full", coalesce=True).join(rec, on=["country", "cell"], how="left").join(prc, on=["country", "cell"], how="left")
out = out.with_columns((pl.col("n_test") / pl.col("n_test").sum().over("country")).round(4).alias("test_share"),
                       (pl.col("n_train") / pl.col("n_train").sum().over("country")).round(4).alias("train_share"))
print(out.select("country", "cell", "n_test", "test_share", "train_share", "true_rate", "val_recall", "val_prec", "test_accept").sort("country", "cell"))
# overall
print(qt.group_by("country").agg(pl.col("y").mean().round(4).alias("train_true_rate")).sort("country").rows())
print(qs.join(m.select("s23_id").unique().with_columns(pl.lit(1).alias("a")), on="s23_id", how="left").group_by("country").agg(pl.col("a").fill_null(0).mean().round(4).alias("test_accept")).sort("country").rows())
print("true pairs per S1 (train):", gt.join(pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id").group_by("country").len().rows(),
      pl.read_parquet(W + "norm_train_s1.parquet", columns=["country"])["country"].value_counts().rows())
print("accepted pairs per S1 (test):", m.join(pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id").group_by("country").len().sort("country").rows())
