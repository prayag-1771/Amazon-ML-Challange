"""Empty-address name rescue (v13). Queries with no address rarely survive blocking (address tokens carry most of
the TF-IDF / e5 signal), so a query can end with no candidate in the cascade set at all. This channel retrieves
the top-10 S1s by name only (char_wb 3-gram TF-IDF over name_core, per country), scores them with a small
LightGBM on name-similarity / ambiguity features and keeps the top-1 pair when pr >= PR_MIN - only for empty-address
queries that have no cascade candidate. Accepted pairs are added to both output files.
US / India only (France is unlabelled; the model is not applied there).

  python -m src.rescue train     # fit on train queries touching no validation S1 -> work/rescue.lgb, validation gain
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .config import WORK_DIR, is_valid_expr

W = str(WORK_DIR) + "/"
K = 10
PR_MIN = 0.9
COUNTRIES = ("India", "US")
F = ["sim", "rk", "gap_best", "gap_2nd", "n_tie", "n_c", "eq", "leq", "is_in", "s1_ncnt", "q_ncnt", "s1_empty",
     "r", "ts", "jw", "len1", "lenq", "r_gap", "r_ntie", "s1_nq"]
P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, verbose=-1, num_threads=14)

def pairs(split):
    """Name-only top-K S1 candidates for every empty-address S2/S3 query, with rescue features."""
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "legal", "addr_empty"]).with_row_index("s1_row")
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "country", "name_core", "legal", "addr_empty"]) for k in (2, 3)]).with_row_index("q_row")
    qe = q.filter(pl.col("addr_empty") == 1)
    parts = []
    for c in COUNTRIES:
        a, b = s1.filter(pl.col("country") == c), qe.filter(pl.col("country") == c)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32)
        X1 = vec.fit_transform(a["name_core"].to_list()).T.tocsr()
        m = sp_matmul_topn(vec.transform(b["name_core"].to_list()), X1, top_n=K, threshold=0.1, sort=True, n_threads=14).tocoo()
        d = pl.DataFrame({"q_row": b["q_row"].to_numpy()[m.row], "s1_row": a["s1_row"].to_numpy()[m.col], "sim": m.data})
        parts.append(d.with_columns(pl.col("sim").rank("ordinal", descending=True).over("q_row").alias("rk")))
    d = pl.concat(parts)
    s1 = s1.join(s1.group_by("country", "name_core").len("s1_ncnt"), on=["country", "name_core"])
    d = d.join(s1.select("s1_row", pl.col("entity_id").alias("s1_id"), "country", pl.col("name_core").alias("n1"), pl.col("legal").alias("l1"), "s1_ncnt", pl.col("addr_empty").alias("s1_empty")), on="s1_row")
    d = d.join(q.select("q_row", pl.col("entity_id").alias("s23_id"), pl.col("name_core").alias("nq"), pl.col("legal").alias("lq")), on="q_row")
    d = d.join(d.group_by("country", "nq").agg(pl.col("q_row").n_unique().alias("q_ncnt")), on=["country", "nq"])
    d = d.with_columns(
        (pl.col("sim") - pl.col("sim").max().over("q_row")).alias("gap_best"),
        (pl.col("sim") - pl.col("sim").top_k(2).min().over("q_row")).alias("gap_2nd"),
        (pl.col("sim") >= pl.col("sim").max().over("q_row") - 0.02).sum().over("q_row").alias("n_tie"),
        pl.len().over("q_row").alias("n_c"), (pl.col("n1") == pl.col("nq")).cast(pl.Int8).alias("eq"),
        pl.when((pl.col("l1") == "") | (pl.col("lq") == "")).then(-1).otherwise((pl.col("l1") == pl.col("lq")).cast(pl.Int8)).alias("leq"),
        (pl.col("country") == "India").cast(pl.Int8).alias("is_in"))
    n1, n2 = d["n1"].to_list(), d["nq"].to_list()
    d = d.with_columns(pl.Series("r", cpdist(n1, n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                       pl.Series("ts", cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                       pl.Series("jw", cpdist(n1, n2, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)),
                       pl.col("n1").str.len_chars().alias("len1"), pl.col("nq").str.len_chars().alias("lenq"))
    return d.with_columns((pl.col("r") - pl.col("r").max().over("q_row")).alias("r_gap"),
                          (pl.col("r") >= pl.col("r").max().over("q_row") - 3).sum().over("q_row").alias("r_ntie"),
                          pl.len().over("s1_row").alias("s1_nq"))

def accept(d, cand):
    """Top-1 rescue pair per query with pr >= PR_MIN, for queries absent from the candidate set `cand`."""
    top = d.sort("pr", descending=True).unique("s23_id", keep="first").filter(pl.col("pr") >= PR_MIN)
    return top.join(cand.select("s23_id").unique(), on="s23_id", how="anti")

def predict(split, cand):
    d = pairs(split)
    d = d.with_columns(pl.Series("pr", lgb.Booster(model_file=W + "rescue.lgb").predict(d.select(F).to_numpy().astype(np.float32))))
    return accept(d, cand).select("s1_id", "s23_id", "country", "pr")

def train():
    from .io_utils import load_ground_truth
    from .metric import macro_f05
    from .stage2 import T
    t0 = time.time()
    gt = load_ground_truth()
    d = pairs("train").join(gt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "s23_id"], how="left")
    d = d.with_columns(pl.col("label").fill_null(0), is_valid_expr("s1_id").alias("vs1"))
    vq = d.filter(pl.col("vs1")).select("q_row").unique()  # queries touching a validation S1 are held out
    tr, va = d.join(vq, on="q_row", how="anti"), d.join(vq, on="q_row", how="semi")
    m = lgb.train(P, lgb.Dataset(tr.select(F).to_numpy().astype(np.float32), tr["label"].to_numpy()), 600)
    m.save_model(W + "rescue.lgb")
    va = va.with_columns(pl.Series("pr", m.predict(va.select(F).to_numpy().astype(np.float32)))).filter(pl.col("vs1"))
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
    gtv = gt.join(s1v.select("s1_id"), on="s1_id", how="semi")
    v = pl.read_parquet(W + "valid_scores_stage2.parquet").join(s1v, on="s1_id")
    top = v.sort("p2", descending=True).unique("s23_id", keep="first")
    base = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T, default=0.75)).select("s1_id", "s23_id")
    add = accept(va, v)
    m0, m1 = (macro_f05(s1v["s1_id"], x, gtv, by=s1v) for x in (base, pl.concat([base, add.select("s1_id", "s23_id")])))
    print(f"rescue: {len(add)} pairs, precision {add['label'].mean():.3f}; F {m0['f05']:.5f} -> {m1['f05']:.5f} "
          f"(US {m0['f05_US']:.5f} -> {m1['f05_US']:.5f}, IN {m0['f05_India']:.5f} -> {m1['f05_India']:.5f}), {time.time() - t0:.0f}s")

if __name__ == "__main__":
    {"train": train}[sys.argv[1]]()
