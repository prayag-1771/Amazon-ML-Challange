"""Empty-address rescue v2: queries with an empty address that the v15 prediction leaves unassigned.
Retrieval = char3 on name_core (as v13) U char3 on noise-stripped name U word tf-idf on noise-stripped name, per country.
  python v10/resc2.py pairs train|test   -> v10/resc2_pairs_{split}.parquet
  python v10/resc2.py fit                -> v10/resc2_s{seed}.lgb, validation delta on top of pred15_v
  python v10/resc2.py apply              -> v10/resc2_test_pr.parquet
"""
import sys, time; sys.path.insert(0, "../business_entity_resolution")
import numpy as np, polars as pl, lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
W = "../work/"
K = 15
F = ["s_c", "s_k", "s_w", "rk_c", "rk_k", "rk_w", "g_c", "g_k", "g_w", "n_c", "eq", "eqk", "leq", "is_in", "s1_ncnt", "s1k_ncnt", "q_ncnt",
     "s1_empty", "r", "rk_r", "ts", "jw", "len1", "lenq", "r_gap", "r_ntie", "s1_nq", "fq_in1", "f1_inq", "nq_tok", "n1_tok", "ftok_eq",
     "min_df_miss1", "min_df_shared", "n_miss1", "n_missq", "comb", "rk_comb", "comb_gap", "comb_gap2"]

def noise_vocab():
    a = pl.read_parquet(W + "v10/addtok.parquet")
    return {c: set(a.filter((pl.col("country") == c) & (pl.col("addr") >= 0.6))["t"].to_list()) for c in ("US", "India")}

def clean(names, noise, vocab1):
    out = []
    for n in names:
        t = [w for w in n.split() if w not in noise and w in vocab1]
        out.append(" ".join(t) if t else n)
    return out

def rowdot(X, Y, ii, jj, B=2_000_000):
    return np.concatenate([np.asarray(X[ii[k:k+B]].multiply(Y[jj[k:k+B]]).sum(1)).ravel() for k in range(0, len(ii), B)]) if len(ii) else np.zeros(0)

def pairs(split):
    t0 = time.time()
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "legal", "addr_empty"]).with_row_index("s1_row")
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "country", "name_core", "legal", "addr_empty"]) for k in (2, 3)]).with_row_index("q_row")
    q = q.filter(pl.col("addr_empty") == 1)
    noise = noise_vocab(); noise["France"] = noise["US"] & set()
    parts, cleaned = [], {}
    for c in s1["country"].unique().to_list():
        a, b = s1.filter(pl.col("country") == c), q.filter(pl.col("country") == c)
        if len(b) == 0: continue
        n1, nq = a["name_core"].fill_null("").to_list(), b["name_core"].fill_null("").to_list()
        vocab1 = set(w for n in n1 for w in n.split())
        nqk = clean(nq, noise.get(c, set()), vocab1); n1k = clean(n1, noise.get(c, set()), vocab1)
        cleaned[c] = (a["s1_row"].to_numpy(), n1k, b["q_row"].to_numpy(), nqk)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.02, sublinear_tf=True, dtype=np.float32).fit(n1)
        vw = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=1, max_df=0.005, sublinear_tf=True, dtype=np.float32).fit(n1)
        A_c, A_k, A_w = vc.transform(n1), vc.transform(n1k), vw.transform(n1)
        B_c, B_k, B_w = vc.transform(nq), vc.transform(nqk), vw.transform(nqk)
        ps = []
        for X, Y in ((B_c, A_c), (B_k, A_k), (B_w, A_w)):
            mm = sp_matmul_topn(X, Y.T.tocsr(), top_n=K, threshold=0.05, sort=True, n_threads=14).tocoo()
            ps.append(np.stack([mm.row, mm.col])); print(c, "view", len(ps), f"{time.time()-t0:.0f}s", flush=True)
        ij = np.unique(np.concatenate(ps, 1), axis=1)
        ii, jj = ij[0], ij[1]
        parts.append(pl.DataFrame({"q_row": b["q_row"].to_numpy()[ii], "s1_row": a["s1_row"].to_numpy()[jj],
                                   "s_c": rowdot(B_c, A_c, ii, jj).astype(np.float32), "s_k": rowdot(B_k, A_k, ii, jj).astype(np.float32),
                                   "s_w": rowdot(B_w, A_w, ii, jj).astype(np.float32), "n1k": np.array(n1k, dtype=object)[jj].tolist(),
                                   "nqk": np.array(nqk, dtype=object)[ii].tolist()}))
        print(c, len(b), "queries", len(ii), "pairs", f"{time.time()-t0:.0f}s", flush=True)
    d = pl.concat(parts).sort("q_row", "s1_row")
    s1 = s1.join(s1.group_by("country", "name_core").agg(pl.len().alias("s1_ncnt")), on=["country", "name_core"])
    d = d.join(s1.select("s1_row", pl.col("entity_id").alias("s1_id"), "country", pl.col("name_core").alias("n1"), pl.col("legal").alias("l1"), "s1_ncnt",
                         pl.col("addr_empty").alias("s1_empty")), on="s1_row")
    d = d.join(q.select("q_row", pl.col("entity_id").alias("s23_id"), pl.col("name_core").alias("nq"), pl.col("legal").alias("lq")), on="q_row")
    d = d.with_columns(pl.len().over("country", "n1k").alias("s1k_ncnt_p"))
    k1 = pl.concat([pl.DataFrame({"s1_row": v[0], "n1k": v[1]}).with_columns(pl.lit(c).alias("country")) for c, v in cleaned.items()])
    k1 = k1.join(k1.group_by("country", "n1k").agg(pl.len().alias("s1k_ncnt")), on=["country", "n1k"]).select("s1_row", "s1k_ncnt")
    d = d.drop("s1k_ncnt_p").join(k1, on="s1_row", how="left").sort("q_row", "s1_row")
    # document frequency of S1 name tokens (per country)
    df1 = s1.select("country", pl.col("name_core").fill_null("").str.split(" ").list.unique().alias("t")).explode("t").group_by("country", "t").agg(pl.len().alias("df"))
    dfm = {(c, t): n for c, t, n in df1.iter_rows()}
    n1l, nql, n1kl, nqkl, cl = d["n1"].to_list(), d["nq"].to_list(), d["n1k"].to_list(), d["nqk"].to_list(), d["country"].to_list()
    fq_in1, f1_inq, nqt, n1t, ftok, mdm, mds, nm1, nmq = ([] for _ in range(9))
    for a1, aq, c in zip(n1kl, nqkl, cl):
        t1, tq = set(a1.split()), set(aq.split())
        sh, m1, mq = t1 & tq, t1 - tq, tq - t1
        fq_in1.append(len(sh) / max(len(tq), 1)); f1_inq.append(len(sh) / max(len(t1), 1)); nqt.append(len(tq)); n1t.append(len(t1))
        ftok.append(int(a1.split()[:1] == aq.split()[:1]))
        mdm.append(min((dfm.get((c, t), 0) for t in m1), default=-1)); mds.append(min((dfm.get((c, t), 0) for t in sh), default=-1))
        nm1.append(len(m1)); nmq.append(len(mq))
    d = d.with_columns(pl.Series("fq_in1", fq_in1, pl.Float32), pl.Series("f1_inq", f1_inq, pl.Float32), pl.Series("nq_tok", nqt, pl.Int16), pl.Series("n1_tok", n1t, pl.Int16),
                       pl.Series("ftok_eq", ftok, pl.Int8), pl.Series("min_df_miss1", mdm, pl.Int32), pl.Series("min_df_shared", mds, pl.Int32),
                       pl.Series("n_miss1", nm1, pl.Int16), pl.Series("n_missq", nmq, pl.Int16))
    d = d.join(d.group_by("country", "nq").agg(pl.col("q_row").n_unique().alias("q_ncnt")), on=["country", "nq"]).sort("q_row", "s1_row")
    d = d.with_columns(pl.Series("r", cpdist(n1l, nql, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                       pl.Series("ts", cpdist(n1l, nql, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                       pl.Series("jw", cpdist(n1l, nql, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)),
                       pl.col("n1").str.len_chars().alias("len1"), pl.col("nq").str.len_chars().alias("lenq"),
                       (pl.col("n1") == pl.col("nq")).cast(pl.Int8).alias("eq"), (pl.col("n1k") == pl.col("nqk")).cast(pl.Int8).alias("eqk"),
                       pl.when((pl.col("l1") == "") | (pl.col("lq") == "")).then(-1).otherwise((pl.col("l1") == pl.col("lq")).cast(pl.Int8)).alias("leq"),
                       (pl.col("country") == "India").cast(pl.Int8).alias("is_in"))
    d = d.with_columns((pl.col("s_c") + pl.col("s_k") + pl.col("s_w") + pl.col("r") / 100).alias("comb"))
    for c in ("c", "k", "w"):
        d = d.with_columns(pl.col("s_" + c).rank("ordinal", descending=True).over("q_row").alias("rk_" + c), (pl.col("s_" + c) - pl.col("s_" + c).max().over("q_row")).alias("g_" + c))
    d = d.with_columns(pl.col("r").rank("ordinal", descending=True).over("q_row").alias("rk_r"), pl.col("comb").rank("ordinal", descending=True).over("q_row").alias("rk_comb"),
                       (pl.col("comb") - pl.col("comb").max().over("q_row")).alias("comb_gap"),
                       (pl.col("comb") - pl.col("comb").top_k(2).min().over("q_row")).alias("comb_gap2"),
                       (pl.col("r") - pl.col("r").max().over("q_row")).alias("r_gap"), (pl.col("r") >= pl.col("r").max().over("q_row") - 3).sum().over("q_row").alias("r_ntie"),
                       pl.len().over("q_row").alias("n_c"), pl.len().over("s1_row").alias("s1_nq"))
    d = d.drop("n1k", "nqk", "l1", "lq")
    print("pairs", len(d), f"{time.time()-t0:.0f}s", flush=True)
    d.write_parquet(W + f"v10/resc2_pairs_{split}.parquet")

P = dict(objective="binary", num_leaves=31, min_data_in_leaf=300, learning_rate=0.05, lambda_l2=5, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=12)

def fit():
    from src.config import is_valid_expr
    from src.metric import macro_f05
    gt = pl.read_parquet(W + "gt_pairs.parquet")
    d = pl.read_parquet(W + "v10/resc2_pairs_train.parquet").join(gt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "s23_id"], how="left")
    d = d.with_columns(pl.col("label").fill_null(0), is_valid_expr("s1_id").alias("vs1"))
    # hold out every query whose truth or any candidate touches a validation S1
    vq = pl.concat([d.filter(pl.col("vs1")).select("s23_id"), gt.filter(is_valid_expr("s1_id")).select("s23_id")]).unique()
    tr, va = d.join(vq, on="s23_id", how="anti"), d.join(vq, on="s23_id", how="semi")
    print("train pairs", len(tr), "pos rate", tr["label"].mean(), "val pairs", len(va), flush=True)
    X = tr.select(F).to_numpy().astype(np.float32); y = tr["label"].to_numpy()
    pr = np.zeros(len(va))
    for s in (1, 2, 3):
        m = lgb.train(dict(P, seed=s), lgb.Dataset(X, y), 500); m.save_model(W + f"v10/resc2_s{s}.lgb")
        pr += m.predict(va.select(F).to_numpy().astype(np.float32)) / 3
    va = va.with_columns(pl.Series("pr", pr))
    va.select("s1_id", "s23_id", "country", "pr", "label", "vs1").write_parquet(W + "v10/resc2_valid_pr.parquet")
    evaluate(va)

def evaluate(va):
    from src.config import is_valid_expr
    from src.metric import macro_f05
    gt = pl.read_parquet(W + "gt_pairs.parquet")
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
    gtv = gt.join(s1v.select("s1_id"), on="s1_id", how="semi")
    base = pl.read_parquet(W + "pred15_v.parquet")
    top = va.sort(["pr", "s1_id"], descending=[True, False]).unique("s23_id", keep="first").join(base.select("s23_id").unique(), on="s23_id", how="anti").filter(pl.col("vs1"))
    b = base.select("s1_id", "s23_id")
    m0 = macro_f05(s1v["s1_id"], b, gtv, by=s1v)
    print(f"base F {m0['f05']:.5f} US {m0['f05_US']:.5f} IN {m0['f05_India']:.5f}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        a = top.filter(pl.col("pr") >= t)
        m1 = macro_f05(s1v["s1_id"], pl.concat([b, a.select("s1_id", "s23_id")]), gtv, by=s1v)
        print(f"t={t}: +{len(a)} (US {a.filter(pl.col('country')=='US').height}, IN {a.filter(pl.col('country')=='India').height}) prec {a['label'].mean():.3f} "
              f"F +{m1['f05']-m0['f05']:.5f} US +{m1['f05_US']-m0['f05_US']:.5f} IN +{m1['f05_India']-m0['f05_India']:.5f}", flush=True)

def apply():
    d = pl.read_parquet(W + "v10/resc2_pairs_test.parquet")
    X = d.select(F).to_numpy().astype(np.float32)
    pr = sum(lgb.Booster(model_file=W + f"v10/resc2_s{s}.lgb").predict(X) for s in (1, 2, 3)) / 3
    d = d.with_columns(pl.Series("pr", pr))
    m = pl.read_csv(W + "sub15_indiaaddr/matching_results.tsv", separator="\t").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    d = d.join(m.select(pl.col("matched_entity_ids").alias("s23_id")).unique(), on="s23_id", how="anti")  # only queries v15 leaves unassigned
    top = d.sort(["pr", "s1_id"], descending=[True, False]).unique("s23_id", keep="first")
    top.select("s1_id", "s23_id", "country", "pr").write_parquet(W + "v10/resc2_test_pr.parquet")
    print(top.group_by("country").agg(pl.len(), *[(pl.col("pr") >= t).sum().alias(f">={t}") for t in (0.6, 0.8, 0.9, 0.95)]))

if __name__ == "__main__":
    {"pairs": lambda: pairs(sys.argv[2]), "fit": fit, "apply": apply, "eval": lambda: evaluate(pl.read_parquet(W + "v10/resc2_valid_pr.parquet"))}[sys.argv[1]]()
