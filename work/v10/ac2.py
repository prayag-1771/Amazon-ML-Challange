"""India addressed no-candidate channel at test-like density: index = all India S1 of the split.
Population: train = singleton queries + queries whose truth pair is not in pruned; test = queries not in cascade.
Channels: per-state address word TF-IDF top-K ∪ name-skeleton char TF-IDF top-K; keep top by combined score. Writes v10/ac2_{split}.parquet with features."""
import sys, time; sys.path.insert(0, "../business_entity_resolution"); sys.path.insert(0, "v10")
import numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from rapidfuzz.distance import JaroWinkler
from translit import skel
W = "../work/"; K = 30
split = sys.argv[1]
s1 = pl.read_parquet(W+f"norm_{split}_s1.parquet", columns=["entity_id","country","name_core","state","addr_core","legal"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32)).filter(pl.col("country")=="India")
q = pl.concat([pl.read_parquet(W+f"norm_{split}_s{k}.parquet", columns=["entity_id","country","name_core","state","addr_core","legal","addr_empty"]) for k in (2,3)]).with_row_index("q_row").with_columns(pl.col("q_row").cast(pl.UInt32)).filter((pl.col("country")=="India")&(pl.col("addr_empty")==0))
if split == "train":
    g = pl.read_parquet(W+"v10/train_gt_rawcov.parquet").filter(pl.col("country")=="India")
    hit = g.filter(pl.col("inpr")).select("q_row").unique()
    q = q.join(hit, on="q_row", how="anti")
else:
    casc = pl.read_parquet(W+"cascade_test_p001.parquet").select(pl.col("s23_id").alias("entity_id")).unique()
    q = q.join(casc, on="entity_id", how="anti")
print(split, "queries", len(q), "S1", len(s1), flush=True)
t0 = time.time()
s1 = s1.with_columns(skel("name_core").alias("sk"), pl.col("addr_core").fill_null("")); q = q.with_columns(skel("name_core").alias("sk"), pl.col("addr_core").fill_null(""))
va = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32).fit(s1["addr_core"].to_list())
vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,3), min_df=2, sublinear_tf=True, dtype=np.float32).fit(s1["sk"].to_list())
QA = va.transform(q["addr_core"].to_list()); SA = va.transform(s1["addr_core"].to_list())
QN = vn.transform(q["sk"].to_list()); SN = vn.transform(s1["sk"].to_list())
qst = q["state"].to_numpy(); sst = s1["state"].to_numpy()
parts = []
for st in np.unique(qst[qst != None]).tolist():
    bi = np.flatnonzero(qst == st); ai = np.flatnonzero(sst == st)
    if len(ai) == 0: continue
    for X, Y in ((QA, SA), (QN, SN)):
        m = sp_matmul_topn(X[bi], Y[ai].T.tocsr(), top_n=K, threshold=0.02, sort=True, n_threads=14).tocoo()
        parts.append(pl.DataFrame({"qi": bi[m.row].astype(np.int32), "si": ai[m.col].astype(np.int32)}))
d = pl.concat(parts).unique()
ii = d["qi"].to_numpy(); jj = d["si"].to_numpy()
def rowdot(X, Y, B=2_000_000):
    return np.concatenate([np.asarray(X[ii[k:k+B]].multiply(Y[jj[k:k+B]]).sum(1)).ravel() for k in range(0, len(ii), B)]).astype(np.float32)
d = d.with_columns(pl.Series("asim", rowdot(QA, SA)), pl.Series("nsim", rowdot(QN, SN)))
d = d.with_columns((pl.col("asim") + pl.col("nsim")).alias("comb"))
for c in ("asim","nsim","comb"):
    d = d.with_columns(pl.col(c).rank("ordinal", descending=True).over("qi").alias("rk_"+c))
print(f"retrieval {time.time()-t0:.0f}s raw pairs {len(d):,}", flush=True)
d = d.filter((pl.col("rk_comb") <= 10) | (pl.col("rk_asim") <= 2) | (pl.col("rk_nsim") <= 2))
d = d.with_columns(pl.Series("q_row", q["q_row"].to_numpy()[d["qi"].to_numpy()]), pl.Series("s1_row", s1["s1_row"].to_numpy()[d["si"].to_numpy()]))
d = d.join(s1.select("s1_row", pl.col("entity_id").alias("s1_id"), pl.col("name_core").alias("n1"), pl.col("sk").alias("k1"), pl.col("addr_core").alias("a1"), pl.col("legal").alias("l1")), on="s1_row")
d = d.join(q.select("q_row", pl.col("entity_id").alias("s23_id"), pl.col("name_core").alias("nq"), pl.col("sk").alias("kq"), pl.col("addr_core").alias("aq"), pl.col("legal").alias("lq")), on="q_row")
tok = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
d = d.with_columns(tok("a1").alias("t1"), tok("aq").alias("tq"))
num = lambda c: pl.col(c).list.eval(pl.element().filter(pl.element().str.contains(r"\d")))
wrd = lambda c: pl.col(c).list.eval(pl.element().filter(~pl.element().str.contains(r"\d")))
d = d.with_columns(num("t1").alias("u1"), num("tq").alias("uq"), wrd("t1").alias("w1"), wrd("tq").alias("wq"))
d = d.with_columns(pl.col("u1").list.set_intersection("uq").list.len().alias("num_sh"), pl.col("uq").list.len().alias("num_q"), pl.col("u1").list.len().alias("num_1"),
                   pl.col("w1").list.set_intersection("wq").list.len().alias("w_sh"), pl.col("wq").list.len().alias("w_q"), pl.col("w1").list.len().alias("w_1"),
                   (pl.col("k1") == pl.col("kq")).cast(pl.Int8).alias("key_eq"), (pl.col("n1") == pl.col("nq")).cast(pl.Int8).alias("eq"),
                   pl.when((pl.col("l1") == "") | (pl.col("lq") == "")).then(-1).otherwise((pl.col("l1") == pl.col("lq")).cast(pl.Int8)).alias("leq")).drop("t1","tq","u1","uq","w1","wq","a1","aq")
d = d.with_columns((pl.col("num_sh") / pl.max_horizontal(pl.col("num_q"), 1)).alias("num_fq"), (pl.col("w_sh") / pl.max_horizontal(pl.col("w_q"), 1)).alias("w_fq"))
n1, n2, k1, k2 = d["n1"].to_list(), d["nq"].to_list(), d["k1"].to_list(), d["kq"].to_list()
d = d.with_columns(pl.Series("r", cpdist(n1, n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                   pl.Series("ts", cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                   pl.Series("jw", cpdist(n1, n2, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)),
                   pl.Series("kr", cpdist(k1, k2, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                   pl.col("n1").str.len_chars().alias("len1"), pl.col("nq").str.len_chars().alias("lenq"))
del n1, n2, k1, k2
d = d.with_columns(pl.len().over("q_row").alias("n_c"), pl.len().over("s1_row").alias("s1_nq"))
for c in ("asim","nsim","comb","kr","num_fq","w_fq"):
    d = d.with_columns((pl.col(c) - pl.col(c).max().over("q_row")).alias(c+"_gap"))
d = d.with_columns((pl.col("comb") - pl.col("comb").top_k(2).min().over("q_row")).alias("comb_gap2"),
                   (pl.col("comb") - pl.col("comb").max().over("s1_row")).alias("comb_s1gap"))
d.drop("qi","si","n1","nq","k1","kq","l1","lq").write_parquet(W+f"v10/ac2_{split}.parquet")
print(f"done {time.time()-t0:.0f}s pairs {len(d):,} q {d['q_row'].n_unique():,}", flush=True)
