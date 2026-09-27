"""Top-1 pairs: label rate of 'word swaps' (a whole name word replaced, not a typo), by country and address agreement.
Train (labelled, US/India) vs test assignment share (France/US/India)."""
import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz

W = "../work/"
LEGAL = set("sa sas sasu sarl eurl sci snc sc ei eirl scop selarl selas gie inc llc corp co ltd pvt private limited llp "
            "company corporation incorporated pllc lp plc and fils associes".split())
cols = ["entity_id", "country", "name_full", "nums"]


def toks(s):
    s = "".join(c if c.isalnum() else " " for c in anyascii(s or "").lower())
    return {t for t in s.split() if len(t) >= 3 and t not in LEGAL}


def kind(a, b):
    ta, tb = toks(a), toks(b)
    xa, xb = ta - tb, tb - ta
    if not xa and not xb:
        return "same"
    # a token is a typo if it is close to some token on the other side
    wa = [t for t in xa if max((fuzz.ratio(t, u) for u in tb), default=0) < 75]
    wb = [t for t in xb if max((fuzz.ratio(t, u) for u in ta), default=0) < 75]
    if wa and wb:
        return "word_swap"
    if wa or wb:
        return "word_addrm"
    return "typo"


def top1(split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=cols)
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=cols) for k in (2, 3)])
    c = pl.read_parquet(W + f"pruned_{split}.parquet", columns=["q_row", "s1_row", "p0_qrank", "s1_id", "s23_id"])
    c = c.filter(pl.col("p0_qrank") == 1)
    A = s1.select("country", pl.col("name_full").alias("n1"), pl.col("nums").alias("m1"))[c["s1_row"].to_numpy()]
    B = q.select(pl.col("name_full").alias("nq"), pl.col("nums").alias("mq"))[c["q_row"].to_numpy()]
    d = pl.concat([c.select("s1_id", "s23_id"), A, B], how="horizontal")
    return d.with_columns(
        pl.when((pl.col("m1") == "") | (pl.col("mq") == "")).then(pl.lit("na"))
          .when(pl.col("mq").str.split(" ").list.contains(pl.col("m1").str.split(" ").list.first())).then(pl.lit("eq"))
          .otherwise(pl.lit("ne")).alias("num"))


tr = top1("train").sample(600_000, seed=0)
gt = pl.read_parquet(W + "gt_pairs.parquet")
print(gt.columns)
gt = gt.select(pl.col(gt.columns[0]).alias("s1_id"), pl.col(gt.columns[1]).alias("s23_id")).with_columns(pl.lit(1).alias("y"))
tr = tr.join(gt, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("y").fill_null(0))
tr = tr.with_columns(pl.Series("kind", [kind(a, b) for a, b in zip(tr["n1"], tr["nq"])]))
pl.Config.set_tbl_rows(60)
print("== train top-1: match rate by kind x house-number agreement ==")
print(tr.group_by("country", "kind", "num").agg(pl.len().alias("n"), pl.col("y").mean().alias("match_rate"))
        .sort("country", "kind", "num"))

te = top1("test")
sc = pl.read_parquet(W + "test_scores_lgb_v4a.parquet").filter(pl.col("p") >= 0.75)
te = te.join(sc, on=["s1_id", "s23_id"], how="semi").sample(900_000, seed=0)
te = te.with_columns(pl.Series("kind", [kind(a, b) for a, b in zip(te["n1"], te["nq"])]))
print("== test assigned top-1 pairs: share by kind x num ==")
print(te.group_by("country", "kind", "num").agg(pl.len().alias("n")).with_columns(
    (pl.col("n") / pl.col("n").sum().over("country")).alias("share")).sort("country", "kind", "num"))
