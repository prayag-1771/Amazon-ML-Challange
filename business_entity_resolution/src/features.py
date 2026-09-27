"""Pairwise + contextual features for (query S2/S3 record, candidate S1 record) pairs.

Country is never used as a feature (open set; France has no labels) - only as a partition key.
Input: candidate frame with q_row, s1_row (+ blocking sims/ranks). Output adds float32 features.
"""
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from .config import WORK_DIR
from .normalize import normalize_source

TEXT_COLS = ["name_full", "name_core", "name_alt", "legal", "addr_full", "addr_core", "state", "nums",
             "alpha_comps", "addr_empty"]


def load_side(split: str):
    """Normalised S1 and query frames with row indices + per-record context columns."""
    s1 = normalize_source(split, 1).with_row_index("s1_row")
    qs = []
    for s in (2, 3):
        d = normalize_source(split, s).with_columns(pl.lit(s, pl.Int8).alias("src"))
        qs.append(d)
    q = pl.concat(qs).with_row_index("q_row")
    q = q.with_columns(pl.col("business_name").str.contains(r"[^\x00-\x7F]").cast(pl.Int8).alias("q_nonascii"))
    # name rarity within S1 of the same country (generic names / chains are ambiguous)
    s1 = s1.with_columns(
        pl.len().over("country", "name_core").alias("s1_name_cnt"),
        pl.len().over("country", "name_core", "state").alias("s1_name_state_cnt"),
    )
    cnt = s1.group_by("country", "name_core").len("q_name_in_s1")
    q = q.join(cnt, on=["country", "name_core"], how="left").with_columns(pl.col("q_name_in_s1").fill_null(0))
    q = q.sort("q_row")
    return s1, q


def _jacc(a: pl.Series, b: pl.Series, sep: str) -> np.ndarray:
    """Token-set Jaccard; -1 when either side is empty."""
    d = pl.DataFrame({"a": a.str.split(sep).list.eval(pl.element().filter(pl.element() != "")),
                      "b": b.str.split(sep).list.eval(pl.element().filter(pl.element() != ""))})
    inter = pl.col("a").list.set_intersection(pl.col("b")).list.len()
    union = pl.col("a").list.set_union(pl.col("b")).list.len()
    empty = (pl.col("a").list.len() == 0) | (pl.col("b").list.len() == 0)
    return d.select(pl.when(empty).then(-1.0).otherwise(inter / union).cast(pl.Float32)).to_series().to_numpy()


def _num_matrix(nums: pl.Series, k: int = 4) -> np.ndarray:
    """First k numeric tokens as a float matrix (nan-padded); 8+ digit tokens (phones, ids) dropped."""
    lst = nums.str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars().is_between(1, 8)))
    m = np.full((len(nums), k), np.nan, dtype=np.float64)
    for j in range(k):
        m[:, j] = lst.list.get(j, null_on_oob=True).cast(pl.Float64).fill_null(np.nan).to_numpy()
    return m


def _num_feats(n1: pl.Series, n2: pl.Series):
    """House-number style features: set Jaccard, first-number match, min relative diff."""
    jac = _jacc(n1, n2, " ")
    a, b = _num_matrix(n1), _num_matrix(n2)
    with np.errstate(all="ignore"):
        eq_a0 = (a[:, :1] == b).any(axis=1)
        eq_b0 = (b[:, :1] == a).any(axis=1)
        diff = np.abs(a[:, :, None] - b[:, None, :]) / np.maximum(np.maximum(a[:, :, None], b[:, None, :]), 1)
        min_rel = np.where(np.isnan(diff), np.inf, diff).reshape(len(a), -1).min(axis=1)
    has = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
    first_eq = np.where(has, (eq_a0 | eq_b0).astype(np.float32), -1).astype(np.float32)
    min_rel = np.where(has & np.isfinite(min_rel), min_rel, -1).astype(np.float32)
    return jac, first_eq, min_rel


def pair_features(cand: pl.DataFrame, s1: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    """cand: q_row, s1_row, ... -> cand + features (row order preserved)."""
    si = cand["s1_row"].to_numpy()
    qi = cand["q_row"].to_numpy()
    A = s1.select(TEXT_COLS + ["s1_name_cnt", "s1_name_state_cnt"])[si]
    B = q.select(TEXT_COLS + ["src", "q_nonascii", "q_name_in_s1"])[qi]
    f = {}

    def cp(scorer, c1, c2=None, **kw):
        return cpdist(A[c1].to_list(), B[c2 or c1].to_list(), scorer=scorer, workers=-1, dtype=np.float32, **kw)

    f["nc_ratio"] = cp(fuzz.ratio, "name_core")
    f["nc_tset"] = cp(fuzz.token_set_ratio, "name_core")
    f["nc_tsort"] = cp(fuzz.token_sort_ratio, "name_core")
    f["nc_partial"] = cp(fuzz.partial_ratio, "name_core")
    f["nc_jw"] = cp(JaroWinkler.normalized_similarity, "name_core")
    f["nf_ratio"] = cp(fuzz.ratio, "name_full")
    # query trade-name part ('X dba Y') vs S1 core name, and nospace comparison ('indigomarketing')
    nalt = cpdist(A["name_core"].to_list(), B["name_alt"].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    f["alt_ratio"] = np.where(B["name_alt"].to_numpy() == "", -1, nalt).astype(np.float32)
    f["nc_nospace"] = cpdist(A["name_core"].str.replace_all(" ", "").to_list(),
                             B["name_core"].str.replace_all(" ", "").to_list(),
                             scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    f["nc_eq"] = (A["name_core"] == B["name_core"]).cast(pl.Float32).to_numpy()
    f["legal_eq"] = np.where((A["legal"] == "").to_numpy() | (B["legal"] == "").to_numpy(), -1,
                             (A["legal"] == B["legal"]).cast(pl.Float32).to_numpy()).astype(np.float32)
    f["nc_len1"] = A["name_core"].str.len_chars().cast(pl.Float32).to_numpy()
    f["nc_len2"] = B["name_core"].str.len_chars().cast(pl.Float32).to_numpy()

    empty = (A["addr_empty"] == 1).to_numpy() | (B["addr_empty"] == 1).to_numpy()
    for name, sc in [("ac_tset", fuzz.token_set_ratio), ("ac_ratio", fuzz.ratio), ("ac_partial", fuzz.partial_ratio)]:
        v = cp(sc, "addr_core")
        f[name] = np.where(empty, -1, v).astype(np.float32)
    f["ac_tsort"] = np.where(empty, -1, cp(fuzz.token_sort_ratio, "addr_core")).astype(np.float32)
    st1, st2 = A["state"].to_numpy(), B["state"].to_numpy()
    f["state_eq"] = np.where((st1 == "") | (st2 == ""), -1, (st1 == st2)).astype(np.float32)
    f["city_jacc"] = _jacc(A["alpha_comps"], B["alpha_comps"], "|")
    f["addr_tok_jacc"] = _jacc(A["addr_core"], B["addr_core"], " ")
    f["num_jacc"], f["num_first_eq"], f["num_min_rel"] = _num_feats(A["nums"], B["nums"])
    f["q_addr_empty"] = B["addr_empty"].cast(pl.Float32).to_numpy()
    f["q_nums_n"] = B["nums"].str.count_matches(r"\d+").cast(pl.Float32).to_numpy()
    f["s1_nums_n"] = A["nums"].str.count_matches(r"\d+").cast(pl.Float32).to_numpy()

    f["src"] = B["src"].cast(pl.Float32).to_numpy()
    f["q_nonascii"] = B["q_nonascii"].cast(pl.Float32).to_numpy()
    f["s1_name_cnt"] = np.log1p(A["s1_name_cnt"].cast(pl.Float32).to_numpy())
    f["s1_name_state_cnt"] = np.log1p(A["s1_name_state_cnt"].cast(pl.Float32).to_numpy())
    f["q_name_in_s1"] = np.log1p(B["q_name_in_s1"].cast(pl.Float32).to_numpy())
    return cand.with_columns(**{k: pl.Series(k, v) for k, v in f.items()})


def context_features(df: pl.DataFrame, score: str) -> pl.DataFrame:
    """Within-query and within-S1 context of a pairwise score column."""
    return df.with_columns(
        pl.len().over("q_row").cast(pl.Float32).alias("q_ncand"),
        pl.col(score).rank("ordinal", descending=True).over("q_row").cast(pl.Float32).alias(f"{score}_qrank"),
        (pl.col(score) - pl.col(score).max().over("q_row")).alias(f"{score}_gap_best"),
        (pl.col(score) - pl.col(score).top_k(2).min().over("q_row")).alias(f"{score}_gap_2nd"),
        (pl.col(score) - pl.col(score).max().over("s1_row")).alias(f"{score}_s1_gap_best"),
        pl.len().over("s1_row").cast(pl.Float32).alias("s1_nq"),
    )


def blocking_feats(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("tf_sim").fill_null(0.0), pl.col("emb_sim").fill_null(0.0),
        pl.col("tf_rank").fill_null(99).cast(pl.Float32), pl.col("emb_rank").fill_null(99).cast(pl.Float32),
    ).with_columns(
        (pl.col("emb_sim") - pl.col("emb_sim").max().over("q_row")).alias("emb_gap_best"),
        (pl.col("tf_sim") - pl.col("tf_sim").max().over("q_row")).alias("tf_gap_best"),
    )


def build_features(cand: pl.DataFrame, split: str, tag: str, chunk: int = 5_000_000) -> None:
    """Compute pair features for a candidate frame in chunks -> work/feat_{tag}_{i}.parquet."""
    s1, q = load_side(split)
    cand = blocking_feats(cand)
    for n, i in enumerate(range(0, len(cand), chunk)):
        t = time.time()
        part = pair_features(cand.slice(i, chunk), s1, q)
        part.write_parquet(WORK_DIR / f"feat_{tag}_{n:03d}.parquet")
        print(f"  feat {tag} chunk {n}: {len(part):,} rows {time.time() - t:.0f}s", flush=True)
