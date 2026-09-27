"""India addressed no-candidate channel (v15; PR_MIN 0.9 -> 0.6 in v16). Some India queries with an address still end with no candidate in the
cascade set - mostly transliterated names ("Shri Ganesh" / "Sree Ganesha") whose address overlaps the true S1 only
partially. For those queries this channel retrieves, per state, the top-K S1s by address word TF-IDF and the top-K by
a phonetic name skeleton (char_wb 2-3 gram TF-IDF over skel(name_core)), scores the union with a 3-seed LightGBM on
address-overlap / name-similarity / ambiguity features and keeps the top-1 pair when pr >= PR_MIN. Accepted pairs are
added to both output files. India only (the population is India-specific; US has no such gap, France is unlabelled).

  python -m src.india_addr train   # fit on train queries whose true pair is not in the pruned set -> work/india_addr_s{1,2,3}.lgb
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
K = 30
PR_MIN = 0.6  # v16: validation sweep 0.5-0.9 peaks at 0.6 (precision 0.93; F0.5 counts the new TP S1s at full weight)
SEEDS = (1, 2, 3)
F = ["asim", "nsim", "comb", "rk_asim", "rk_nsim", "rk_comb", "num_sh", "num_q", "num_1", "w_sh", "w_q", "w_1", "key_eq", "eq", "leq",
     "num_fq", "w_fq", "r", "ts", "jw", "kr", "len1", "lenq", "n_c", "s1_nq", "asim_gap", "nsim_gap", "comb_gap", "kr_gap", "num_fq_gap",
     "w_fq_gap", "comb_gap2", "comb_s1gap"]
P = dict(objective="binary", num_leaves=31, min_data_in_leaf=300, learning_rate=0.05, lambda_l2=5, feature_fraction=0.7,
         bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=14)

def skel(col):
    """Phonetic skeleton of a name: merge common transliteration variants, drop vowels / h, squeeze repeats."""
    e = pl.col(col).fill_null("").str.to_lowercase()
    for a, b in [(r"[^a-z0-9 ]", ""), ("ph", "f"), ("ck", "k"), (r"c([eiy])", "s$1"), ("c", "k"), ("q", "k"), ("x", "ks"), ("z", "s"),
                 ("w", "v"), ("h", ""), (r"[aeiouy]", ""), ("d", "t"), ("b", "p"), ("g", "k"), ("v", "f"), ("j", "k"), ("m", "n")]:
        e = e.str.replace_all(a, b)
    for ch in "bcdfgklnprstz":
        e = e.str.replace_all(ch + "+", ch)
    return e.str.replace_all(r" +", " ").str.strip_chars()

def pairs(split, exclude):
    """Address ∪ name-skeleton top-K S1 candidates for India addressed queries not in `exclude` (s23_id), with features."""
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "state", "addr_core", "legal"]) \
        .with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32)).filter(pl.col("country") == "India")
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "country", "name_core", "state", "addr_core", "legal", "addr_empty"])
                   for k in (2, 3)]).with_row_index("q_row").with_columns(pl.col("q_row").cast(pl.UInt32)) \
        .filter((pl.col("country") == "India") & (pl.col("addr_empty") == 0))
    q = q.join(exclude.select(pl.col("s23_id").alias("entity_id")).unique(), on="entity_id", how="anti")
    s1 = s1.with_columns(skel("name_core").alias("sk"), pl.col("addr_core").fill_null(""))
    q = q.with_columns(skel("name_core").alias("sk"), pl.col("addr_core").fill_null(""))
    va = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32).fit(s1["addr_core"].to_list())
    vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2, sublinear_tf=True, dtype=np.float32).fit(s1["sk"].to_list())
    QA, SA = va.transform(q["addr_core"].to_list()), va.transform(s1["addr_core"].to_list())
    QN, SN = vn.transform(q["sk"].to_list()), vn.transform(s1["sk"].to_list())
    qst, sst = q["state"].to_numpy(), s1["state"].to_numpy()
    parts = []
    for st in np.unique(qst[qst != None]).tolist():  # noqa: E711 - numpy object array
        bi, ai = np.flatnonzero(qst == st), np.flatnonzero(sst == st)
        if len(ai) == 0:
            continue
        for X, Y in ((QA, SA), (QN, SN)):
            m = sp_matmul_topn(X[bi], Y[ai].T.tocsr(), top_n=K, threshold=0.02, sort=True, n_threads=14).tocoo()
            parts.append(pl.DataFrame({"qi": bi[m.row].astype(np.int32), "si": ai[m.col].astype(np.int32)}))
    d = pl.concat(parts).unique().sort("qi", "si")  # fixed order: ordinal ranks break ties deterministically
    ii, jj = d["qi"].to_numpy(), d["si"].to_numpy()
    def rowdot(X, Y, B=2_000_000):
        return np.concatenate([np.asarray(X[ii[k:k + B]].multiply(Y[jj[k:k + B]]).sum(1)).ravel() for k in range(0, len(ii), B)]).astype(np.float32)
    d = d.with_columns(pl.Series("asim", rowdot(QA, SA)), pl.Series("nsim", rowdot(QN, SN)))
    d = d.with_columns((pl.col("asim") + pl.col("nsim")).alias("comb"))
    d = d.with_columns([pl.col(c).rank("ordinal", descending=True).over("qi").alias("rk_" + c) for c in ("asim", "nsim", "comb")])
    d = d.filter((pl.col("rk_comb") <= 10) | (pl.col("rk_asim") <= 2) | (pl.col("rk_nsim") <= 2))
    d = d.with_columns(pl.Series("q_row", q["q_row"].to_numpy()[d["qi"].to_numpy()]), pl.Series("s1_row", s1["s1_row"].to_numpy()[d["si"].to_numpy()]))
    d = d.join(s1.select("s1_row", pl.col("entity_id").alias("s1_id"), pl.col("name_core").alias("n1"), pl.col("sk").alias("k1"),
                         pl.col("addr_core").alias("a1"), pl.col("legal").alias("l1")), on="s1_row")
    d = d.join(q.select("q_row", pl.col("entity_id").alias("s23_id"), pl.col("name_core").alias("nq"), pl.col("sk").alias("kq"),
                        pl.col("addr_core").alias("aq"), pl.col("legal").alias("lq")), on="q_row")
    tok = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    num = lambda c: pl.col(c).list.eval(pl.element().filter(pl.element().str.contains(r"\d")))
    wrd = lambda c: pl.col(c).list.eval(pl.element().filter(~pl.element().str.contains(r"\d")))
    d = d.with_columns(tok("a1").alias("t1"), tok("aq").alias("tq"))
    d = d.with_columns(num("t1").alias("u1"), num("tq").alias("uq"), wrd("t1").alias("w1"), wrd("tq").alias("wq"))
    d = d.with_columns(pl.col("u1").list.set_intersection("uq").list.len().alias("num_sh"), pl.col("uq").list.len().alias("num_q"),
                       pl.col("u1").list.len().alias("num_1"), pl.col("w1").list.set_intersection("wq").list.len().alias("w_sh"),
                       pl.col("wq").list.len().alias("w_q"), pl.col("w1").list.len().alias("w_1"),
                       (pl.col("k1") == pl.col("kq")).cast(pl.Int8).alias("key_eq"), (pl.col("n1") == pl.col("nq")).cast(pl.Int8).alias("eq"),
                       pl.when((pl.col("l1") == "") | (pl.col("lq") == "")).then(-1).otherwise((pl.col("l1") == pl.col("lq")).cast(pl.Int8)).alias("leq")) \
        .drop("t1", "tq", "u1", "uq", "w1", "wq", "a1", "aq")
    d = d.with_columns((pl.col("num_sh") / pl.max_horizontal(pl.col("num_q"), 1)).alias("num_fq"), (pl.col("w_sh") / pl.max_horizontal(pl.col("w_q"), 1)).alias("w_fq"))
    n1, n2, k1, k2 = d["n1"].to_list(), d["nq"].to_list(), d["k1"].to_list(), d["kq"].to_list()
    d = d.with_columns(pl.Series("r", cpdist(n1, n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                       pl.Series("ts", cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                       pl.Series("jw", cpdist(n1, n2, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)),
                       pl.Series("kr", cpdist(k1, k2, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                       pl.col("n1").str.len_chars().alias("len1"), pl.col("nq").str.len_chars().alias("lenq"))
    d = d.with_columns(pl.len().over("q_row").alias("n_c"), pl.len().over("s1_row").alias("s1_nq"))
    d = d.with_columns([(pl.col(c) - pl.col(c).max().over("q_row")).alias(c + "_gap") for c in ("asim", "nsim", "comb", "kr", "num_fq", "w_fq")])
    d = d.with_columns((pl.col("comb") - pl.col("comb").top_k(2).min().over("q_row")).alias("comb_gap2"),
                       (pl.col("comb") - pl.col("comb").max().over("s1_row")).alias("comb_s1gap"))
    return d.drop("qi", "si", "n1", "nq", "k1", "kq", "l1", "lq")

def score(d):
    X = d.select(F).to_numpy().astype(np.float32)
    return d.with_columns(pl.Series("pr", np.mean([lgb.Booster(model_file=W + f"india_addr_s{s}.lgb").predict(X) for s in SEEDS], axis=0)))

def accept(d, cand):
    """Top-1 pair per query with pr >= PR_MIN, for queries absent from the candidate set `cand`."""
    top = d.sort("pr", descending=True).unique("s23_id", keep="first").filter(pl.col("pr") >= PR_MIN)
    return top.join(cand.select("s23_id").unique(), on="s23_id", how="anti")

def predict(split, cand):
    return accept(score(pairs(split, cand)), cand).select("s1_id", "s23_id", "pr")

def train():
    from .io_utils import load_ground_truth
    from .metric import macro_f05
    from .stage2 import T
    t0 = time.time()
    gt = load_ground_truth()
    hit = gt.join(pl.read_parquet(W + "pruned_train.parquet", columns=["s1_id", "s23_id"]), on=["s1_id", "s23_id"], how="semi")
    d = pairs("train", hit)  # singletons and queries whose true pair blocking missed
    d = d.join(gt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "s23_id"], how="left")
    d = d.with_columns(pl.col("label").fill_null(0), is_valid_expr("s1_id").alias("vs1"))
    vq = d.filter(pl.col("vs1")).select("q_row").unique()  # queries touching a validation S1 are held out
    tr, va = d.join(vq, on="q_row", how="anti"), d.join(vq, on="q_row", how="semi")
    X, y = tr.select(F).to_numpy().astype(np.float32), tr["label"].to_numpy()
    for s in SEEDS:
        lgb.train({**P, "seed": s}, lgb.Dataset(X, y), 500).save_model(W + f"india_addr_s{s}.lgb")
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
    gtv = gt.join(s1v.select("s1_id"), on="s1_id", how="semi")
    v = pl.read_parquet(W + "valid_scores_stage2.parquet").join(s1v, on="s1_id")
    base = v.sort("p2", descending=True).unique("s23_id", keep="first").filter(pl.col("p2") >= pl.col("country").replace_strict(T, default=0.75)).select("s1_id", "s23_id")
    cand = pl.read_parquet(W + "cascade_valid_p001.parquet").select("s23_id")
    add = accept(score(va), cand).filter(pl.col("vs1")).join(base, on="s23_id", how="anti")
    m0, m1 = (macro_f05(s1v["s1_id"], x, gtv, by=s1v) for x in (base, pl.concat([base, add.select("s1_id", "s23_id")])))
    print(f"india_addr: {len(add)} pairs, precision {add['label'].mean():.3f}; F {m0['f05']:.5f} -> {m1['f05']:.5f} "
          f"(IN {m0['f05_India']:.5f} -> {m1['f05_India']:.5f}), {time.time() - t0:.0f}s")

if __name__ == "__main__":
    {"train": train}[sys.argv[1]]()
