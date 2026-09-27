"""v17 France cell audit (label-free transfer from US/India validation).

Every query's top-1 pair (stage-2 p2) is put in a cell by how it differs from its S1:
  name   eq | add1 | drop1 | swap1 | other          (name_core tokens)
  num    eq | q_missing | s1_missing | both_missing | up | down | far   (first house number, shift <= 25)
  legal  same | add | drop | chg                    (legal-form tokens)
  street hi | lo                                    (street-word overlap >= 0.5, city words of the S1 removed)
Empty-address queries are left out (separate channel).

Validation (US/India, labels): cell purity (true-match rate) and v15 acceptance.
Test France: v15 acceptance. A cell is a France flip candidate when it is >= 99.9% pure on validation
yet France rejects a much larger share of it than validation does.

  python v17/fr_cells.py        (run from work/)  -> v17/fr_cells_val.parquet, v17/fr_cells_fr.parquet, v17/fr_top1.parquet
"""
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parent / "business_entity_resolution"))
import polars as pl
from src.config import is_valid_expr

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(240)
W = str(WORK) + "/"
OUT = WORK / "v17"
COLS = ["entity_id", "name_core", "legal", "nums", "addr_core", "alpha_comps"]
CELL = ["name", "num", "legal", "street"]


def read_pairs(path, col):
    t = pl.read_csv(path, separator="\t", quote_char=None, schema_overrides={col: pl.Utf8}).with_columns(pl.col(col).fill_null(""))
    return t.with_columns(pl.col(col).str.split(",")).explode(col).filter(pl.col(col) != "").select(
        pl.col("source1_entity_id").alias("s1_id"), pl.col(col).alias("s23_id"))


def cells(top, split):
    """top: s1_id, s23_id, ... (one row per query) -> + cell columns. Loads only the rows it needs."""
    ids1, idq = top.select("s1_id").unique().lazy(), top.select("s23_id").unique().lazy()
    s1 = (pl.scan_parquet(W + f"norm_{split}_s1.parquet").select(COLS).rename(
        {"entity_id": "s1_id", "name_core": "n1", "legal": "l1", "nums": "u1", "addr_core": "a1", "alpha_comps": "c1"})
        .join(ids1, on="s1_id", how="semi").collect())
    q = (pl.concat([pl.scan_parquet(W + f"norm_{split}_s{k}.parquet").select(COLS + ["addr_empty"]) for k in (2, 3)]).rename(
        {"entity_id": "s23_id", "name_core": "nq", "legal": "lq", "nums": "uq", "addr_core": "aq", "alpha_comps": "cq"})
        .join(idq, on="s23_id", how="semi").collect())
    x = top.join(s1, on="s1_id").join(q, on="s23_id").filter(pl.col("addr_empty") == 0)
    words = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    street = lambda a: pl.col(a).fill_null("").str.split(" ").list.set_difference(
        pl.col("c1").fill_null("").str.replace_all(r"\|", " ").str.split(" ")).list.eval(
        pl.element().filter((pl.element() != "") & ~pl.element().str.contains(r"\d")))
    x = x.with_columns(words("n1").alias("w1"), words("nq").alias("wq"), street("a1").alias("t1"), street("aq").alias("tq"),
                       pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False).alias("h1"),
                       pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False).alias("hq"))
    x = x.with_columns(pl.col("wq").list.set_difference("w1").list.len().alias("na"), pl.col("w1").list.set_difference("wq").list.len().alias("nd"),
                       (pl.col("t1").list.set_intersection("tq").list.len() / pl.max_horizontal(pl.col("t1").list.len(), pl.col("tq").list.len(), 1)).alias("st_ov"),
                       (pl.col("hq") - pl.col("h1")).alias("dh"))
    name = (pl.when((pl.col("na") == 0) & (pl.col("nd") == 0)).then(pl.lit("eq")).when((pl.col("na") == 1) & (pl.col("nd") == 0)).then(pl.lit("add1"))
            .when((pl.col("na") == 0) & (pl.col("nd") == 1)).then(pl.lit("drop1")).when((pl.col("na") == 1) & (pl.col("nd") == 1)).then(pl.lit("swap1"))
            .otherwise(pl.lit("other")))
    num = (pl.when(pl.col("h1").is_null() & pl.col("hq").is_null()).then(pl.lit("both_missing"))
           .when(pl.col("hq").is_null()).then(pl.lit("q_missing")).when(pl.col("h1").is_null()).then(pl.lit("s1_missing"))
           .when(pl.col("dh") == 0).then(pl.lit("eq")).when(pl.col("dh").is_between(1, 25)).then(pl.lit("up"))
           .when(pl.col("dh").is_between(-25, -1)).then(pl.lit("down")).otherwise(pl.lit("far")))
    l1, lq = pl.col("l1").fill_null(""), pl.col("lq").fill_null("")
    legal = pl.when(l1 == lq).then(pl.lit("same")).when(l1 == "").then(pl.lit("add")).when(lq == "").then(pl.lit("drop")).otherwise(pl.lit("chg"))
    street_b = pl.when(pl.col("st_ov") >= 0.5).then(pl.lit("hi")).otherwise(pl.lit("lo"))
    keep = [c for c in top.columns] + ["n1", "nq"]
    return x.select(*keep, name.alias("name"), num.alias("num"), legal.alias("legal"), street_b.alias("street"))


def main():
    # ---------------- validation: US/India
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}).filter(is_valid_expr("s1_id"))
    pred = pl.read_parquet(W + "pred15_v.parquet").select("s23_id").unique().with_columns(pl.lit(True).alias("acc"))
    v = pl.read_parquet(W + "valid_scores_stage2.parquet", columns=["s1_id", "s23_id", "p2", "label"]).join(s1v, on="s1_id")
    vt = cells(v.sort("p2", descending=True).unique("s23_id", keep="first"), "train").join(pred, on="s23_id", how="left").with_columns(pl.col("acc").fill_null(False))
    val = vt.group_by(CELL).agg(pl.len().alias("n_val"), pl.col("label").mean().alias("purity"), pl.col("acc").mean().alias("val_acc"))
    val.write_parquet(OUT / "fr_cells_val.parquet")
    del v, vt
    # ---------------- test: France under v15
    m15 = read_pairs(W + "sub15_indiaaddr/matching_results.tsv", "matched_entity_ids")
    frs1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}).filter(pl.col("country") == "France")
    t = pl.read_parquet(W + "test_scores_stage2_v15.parquet", columns=["s1_id", "s23_id", "p", "p2"]).join(frs1.select("s1_id"), on="s1_id", how="semi")
    top = t.sort("p2", descending=True).unique("s23_id", keep="first")
    # v15 decision for the query: matched to this S1 / matched elsewhere (rescue etc.) / unmatched
    acc = m15.join(top.select("s23_id"), on="s23_id", how="semi").rename({"s1_id": "m_s1"})
    ft = cells(top, "test").join(acc, on="s23_id", how="left").with_columns(
        (pl.col("m_s1") == pl.col("s1_id")).fill_null(False).alias("acc"), pl.col("m_s1").is_not_null().alias("q_matched"))
    ft.write_parquet(OUT / "fr_top1.parquet")
    fr = ft.group_by(CELL).agg(pl.len().alias("n_fr"), pl.col("acc").mean().alias("fr_acc"),
                               (~pl.col("q_matched") & (pl.col("p2") >= 0.01)).sum().alias("fr_rej_p01"))
    fr.write_parquet(OUT / "fr_cells_fr.parquet")
    a = val.join(fr, on=CELL, how="full", coalesce=True).with_columns(
        ((1 - pl.col("fr_acc")) / (1 - pl.col("val_acc")).clip(1e-4, None)).alias("rej_ratio"))
    show = a.filter((pl.col("n_val") >= 300) & (pl.col("n_fr") >= 100)).sort("n_fr", descending=True)
    print(show.select(*CELL, "n_val", pl.col("purity").round(4), pl.col("val_acc").round(4), "n_fr", pl.col("fr_acc").round(4),
                      "fr_rej_p01", pl.col("rej_ratio").round(1)).head(60))
    print("\nFLIP candidates (val purity >= 0.999, n_val >= 300, France rejects >= 5x more):")
    print(a.filter((pl.col("purity") >= 0.999) & (pl.col("n_val") >= 300) & (pl.col("rej_ratio") >= 5) & (pl.col("fr_rej_p01") > 0))
          .select(*CELL, "n_val", pl.col("purity").round(4), pl.col("val_acc").round(4), "n_fr", pl.col("fr_acc").round(4), "fr_rej_p01").sort("fr_rej_p01", descending=True))
    print("\nREJECT candidates (val purity <= 0.02, n_val >= 300, France accepts more than validation):")
    print(a.filter((pl.col("purity") <= 0.02) & (pl.col("n_val") >= 300) & (pl.col("fr_acc") > pl.col("val_acc") + 0.01))
          .select(*CELL, "n_val", pl.col("purity").round(4), pl.col("val_acc").round(4), "n_fr", pl.col("fr_acc").round(4)).sort("n_fr", descending=True))


if __name__ == "__main__":
    main()
