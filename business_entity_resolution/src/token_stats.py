"""Label-free name-token statistics, computed separately on each split (so they transfer to France).

For every query's top-1 candidate after pruning we look at the name tokens present on only one side
("extra" tokens) and at whether the first house number of S1 appears among the query's numbers.
Per (country, side, token) the rate of house-number agreement is a strong proxy for how often that
extra token marks a *different* entity (train: corr 0.98 with the true match rate). Tokens such as
'holding', 'participations', 'group', 'north' have agreement ~0.02 (distractor markers); spelling or
legal-form variants have ~0.7. No labels are used, so the same table is built on the test split.

A second table keys the statistic by context as well: kind (a = token added while the other side has no
extra tokens, s = swapped against other tokens) and position in the name (P first, S last, M middle).
The same word can be a distractor marker in one context and harmless noise in another (e.g. an added
mid-name 'groupe' vs a swapped suffix 'groupe'); train AUC of rel vs truth: xq 0.834 -> 0.850, x1 0.611 -> 0.729.
"""
import numpy as np
import polars as pl

from .config import WORK_DIR

M_SMOOTH = 20  # pseudo-count towards the country prior


def _extra_tokens(a: pl.Expr, b: pl.Expr):
    ta = a.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    tb = b.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    return tb.list.set_difference(ta).alias("xq"), ta.list.set_difference(tb).alias("x1")


def _numeq(na: pl.Expr, nb: pl.Expr) -> pl.Expr:
    return (pl.when((na == "") | (nb == "")).then(None)
            .otherwise(nb.str.split(" ").list.contains(na.str.split(" ").list.first()).cast(pl.Float32)))


def _toks(e: pl.Expr) -> pl.Expr:
    return e.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _explode_ctx(d: pl.DataFrame, side: str, keep: list[str]) -> pl.DataFrame:
    """One row per extra token of `side` with its kind (a|s) and position (P|S|M) in its own name."""
    other = "x1" if side == "xq" else "xq"
    own = "tb" if side == "xq" else "ta"
    e = d.select(*keep, pl.when(pl.col(other).list.len() == 0).then(pl.lit("a")).otherwise(pl.lit("s")).alias("kind"),
                 pl.col(own).alias("toks"), pl.col(side).alias("tok")).explode("tok").drop_nulls("tok")
    return e.with_columns(pl.when(pl.col("toks").list.first() == pl.col("tok")).then(pl.lit("P"))
                          .when(pl.col("toks").list.last() == pl.col("tok")).then(pl.lit("S"))
                          .otherwise(pl.lit("M")).alias("pos")).drop("toks")


def _pairs_ctx(A: pl.DataFrame, nb: pl.Series) -> pl.DataFrame:
    return A.with_columns(nb.alias("nb")).with_columns(_toks(pl.col("name_full")).alias("ta"), _toks(pl.col("nb")).alias("tb")).with_columns(
        pl.col("tb").list.set_difference(pl.col("ta")).alias("xq"), pl.col("ta").list.set_difference(pl.col("tb")).alias("x1"))


def token_table_ctx(split: str, s1: pl.DataFrame, q: pl.DataFrame, force: bool = False) -> pl.DataFrame:
    """country, side, tok, kind, pos, krel (smoothed numeq rate / country prior), klfreq."""
    path = WORK_DIR / f"tokstat2_{split}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    c = pl.read_parquet(WORK_DIR / f"pruned_{split}.parquet", columns=["q_row", "s1_row", "p0_qrank"])
    c = c.filter(pl.col("p0_qrank") == 1)
    A = s1.select("country", "name_full", "nums")[c["s1_row"].to_numpy()]
    B = q.select("name_full", "nums")[c["q_row"].to_numpy()]
    d = _pairs_ctx(A.with_columns(B["nums"].alias("mb")), B["name_full"]).with_columns(_numeq(pl.col("nums"), pl.col("mb")).alias("numeq"))
    out = []
    for side in ("xq", "x1"):
        e = _explode_ctx(d, side, ["country", "numeq"])
        prior = e.group_by("country").agg(pl.col("numeq").mean().alias("prior"))
        g = e.group_by("country", "tok", "kind", "pos").agg(pl.len().alias("n"), pl.col("numeq").sum().alias("s"),
                                                            pl.col("numeq").count().alias("k"))
        g = g.join(prior, on="country").with_columns(
            ((pl.col("s") + M_SMOOTH * pl.col("prior")) / (pl.col("k") + M_SMOOTH) / pl.col("prior")).alias("krel"))
        out.append(g.select("country", pl.lit(side).alias("side"), "tok", "kind", "pos", pl.col("krel").cast(pl.Float32),
                            pl.col("n").log1p().cast(pl.Float32).alias("klfreq")))
    tab = pl.concat(out)
    tab.write_parquet(path)
    return tab


def token_table(split: str, s1: pl.DataFrame, q: pl.DataFrame, force: bool = False) -> pl.DataFrame:
    """country, side (xq|x1), tok, rate (smoothed), rel (rate / country prior), lfreq."""
    path = WORK_DIR / f"tokstat_{split}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    c = pl.read_parquet(WORK_DIR / f"pruned_{split}.parquet", columns=["q_row", "s1_row", "p0_qrank"])
    c = c.filter(pl.col("p0_qrank") == 1)
    A = s1.select("country", "name_full", "nums")[c["s1_row"].to_numpy()]
    B = q.select(pl.col("name_full").alias("nb"), pl.col("nums").alias("mb"))[c["q_row"].to_numpy()]
    d = pl.concat([A, B], how="horizontal").select(
        "country", _numeq(pl.col("nums"), pl.col("mb")).alias("numeq"), *_extra_tokens(pl.col("name_full"), pl.col("nb")))
    out = []
    for side in ("xq", "x1"):
        e = d.select("country", "numeq", pl.col(side).alias("tok")).explode("tok").drop_nulls("tok")
        prior = e.group_by("country").agg(pl.col("numeq").mean().alias("prior"))
        g = e.group_by("country", "tok").agg(pl.len().alias("n"), pl.col("numeq").sum().alias("s"),
                                             pl.col("numeq").count().alias("k"))
        g = g.join(prior, on="country").with_columns(
            ((pl.col("s") + M_SMOOTH * pl.col("prior")) / (pl.col("k") + M_SMOOTH)).alias("rate"))
        out.append(g.select("country", pl.lit(side).alias("side"), "tok", pl.col("rate").cast(pl.Float32),
                            (pl.col("rate") / pl.col("prior")).cast(pl.Float32).alias("rel"),
                            pl.col("n").log1p().cast(pl.Float32).alias("lfreq"), pl.col("prior").cast(pl.Float32)))
    tab = pl.concat(out)
    tab.write_parquet(path)
    return tab


def token_features(cand: pl.DataFrame, split: str, s1: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    """Adds per-pair aggregates of the extra-token statistics (row order preserved)."""
    tab = token_table(split, s1, q)
    A = s1.select("country", "name_full")[cand["s1_row"].to_numpy()]
    B = q.select(pl.col("name_full").alias("nb"))[cand["q_row"].to_numpy()]
    d = pl.concat([A, B], how="horizontal").with_row_index("i").select(
        "i", "country", *_extra_tokens(pl.col("name_full"), pl.col("nb")))
    f = {}
    for side in ("xq", "x1"):
        t = tab.filter(pl.col("side") == side).drop("side")
        prior = t.group_by("country").agg(pl.col("prior").first())
        e = d.select("i", "country", pl.col(side).alias("tok")).explode("tok").drop_nulls("tok")
        e = e.join(t.drop("prior"), on=["country", "tok"], how="left").join(prior, on="country", how="left")
        e = e.with_columns(pl.col("rate").fill_null(pl.col("prior")), pl.col("rel").fill_null(1.0), pl.col("lfreq").fill_null(0.0))
        g = e.group_by("i").agg(pl.len().alias("n"), pl.col("rate").min().alias("rmin"), pl.col("rate").mean().alias("rmean"),
                                pl.col("rel").min().alias("relmin"), pl.col("rel").mean().alias("relmean"),
                                pl.col("lfreq").max().alias("fmax"))
        g = pl.DataFrame({"i": np.arange(len(cand), dtype=np.uint32)}).join(g.with_columns(pl.col("i").cast(pl.UInt32)), on="i", how="left").sort("i")
        f[f"{side}_n"] = g["n"].fill_null(0).cast(pl.Float32)
        for k in ("rmin", "rmean", "relmin", "relmean", "fmax"):
            f[f"{side}_{k}"] = g[k].fill_null(-1.0).cast(pl.Float32)
    tab2 = token_table_ctx(split, s1, q)
    d2 = _pairs_ctx(A, B["nb"]).with_row_index("i")
    for side in ("xq", "x1"):
        t = tab2.filter(pl.col("side") == side).drop("side")
        e = _explode_ctx(d2, side, ["i", "country"]).join(t, on=["country", "tok", "kind", "pos"], how="left")
        e = e.with_columns(pl.col("krel").fill_null(1.0), pl.col("klfreq").fill_null(0.0))
        g = e.group_by("i").agg(pl.col("krel").min().alias("krelmin"), pl.col("krel").mean().alias("krelmean"),
                                pl.col("klfreq").min().alias("kfmin"))
        g = pl.DataFrame({"i": np.arange(len(cand), dtype=np.uint32)}).join(g.with_columns(pl.col("i").cast(pl.UInt32)), on="i", how="left").sort("i")
        for k in ("krelmin", "krelmean", "kfmin"):
            f[f"{side}_{k}"] = g[k].fill_null(-1.0).cast(pl.Float32)
    return cand.with_columns(**{k: v.alias(k) for k, v in f.items()})
