"""Address-token retrieval channel for India addressed queries with no cascade candidate. Recall probe on validation missed pairs."""
import sys, time; sys.path.insert(0, "../business_entity_resolution"); sys.path.insert(0, "v10")
import numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from translit import skel
W = "../work/"
K = 50
split = sys.argv[1] if len(sys.argv) > 1 else "train"
s1 = pl.read_parquet(W+f"norm_{split}_s1.parquet", columns=["entity_id","country","name_core","state","addr_core"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32)).filter(pl.col("country")=="India")
q = pl.concat([pl.read_parquet(W+f"norm_{split}_s{k}.parquet", columns=["entity_id","country","name_core","state","addr_core","addr_empty"]) for k in (2,3)]).with_row_index("q_row").with_columns(pl.col("q_row").cast(pl.UInt32)).filter((pl.col("country")=="India")&(pl.col("addr_empty")==0))
casc = pl.read_parquet(W+("cascade_valid_p001.parquet" if split == "train" else "cascade_test_p001.parquet"))
q = q.join(casc.select(pl.col("s23_id").alias("entity_id")).unique(), on="entity_id", how="anti")
print("India addressed no-cascade queries", len(q), flush=True)
if split == "train":
    from src.config import is_valid_expr
    # restrict to queries whose truth (if any) is validation S1: drop queries with a train-S1 truth
    gt = pl.read_parquet(W+"gt_pairs.parquet")
    trq = gt.filter(~is_valid_expr("s1_id")).select(pl.col("s23_id").alias("entity_id")).unique()
    q = q.join(trq, on="entity_id", how="anti")
    s1 = s1.filter(is_valid_expr("entity_id"))
    print("eval queries", len(q), "val S1", len(s1), flush=True)
t0 = time.time()
# address channel: word unigram+bigram tf-idf within state; name-skeleton channel: char 2-3
va = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1,1), min_df=1, sublinear_tf=True, dtype=np.float32).fit(s1["addr_core"].fill_null("").to_list())
s1 = s1.with_columns(skel("name_core").alias("sk")); q = q.with_columns(skel("name_core").alias("sk"))
vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,3), min_df=2, sublinear_tf=True, dtype=np.float32).fit(s1["sk"].to_list())
parts = []
for st in q["state"].unique().to_list():
    a = s1.filter(pl.col("state")==st); b = q.filter(pl.col("state")==st)
    if len(a) == 0 or len(b) == 0: continue
    A1 = va.transform(a["addr_core"].fill_null("").to_list()); B1 = va.transform(b["addr_core"].fill_null("").to_list())
    N1 = vn.transform(a["sk"].to_list()); M1 = vn.transform(b["sk"].to_list())
    ma = sp_matmul_topn(B1, A1.T.tocsr(), top_n=K, threshold=0.02, sort=True, n_threads=14).tocoo()
    mn = sp_matmul_topn(M1, N1.T.tocsr(), top_n=K, threshold=0.02, sort=True, n_threads=14).tocoo()
    parts.append(pl.DataFrame({"q_row": b["q_row"].to_numpy()[ma.row], "s1_row": a["s1_row"].to_numpy()[ma.col], "asim": ma.data, "nsim": np.full(len(ma.data), np.nan, np.float32)}))
    parts.append(pl.DataFrame({"q_row": b["q_row"].to_numpy()[mn.row], "s1_row": a["s1_row"].to_numpy()[mn.col], "asim": np.full(len(mn.data), np.nan, np.float32), "nsim": mn.data}))
d = pl.concat(parts).group_by("q_row","s1_row").agg(pl.col("asim").max(), pl.col("nsim").max())
# fill missing sim side via exact recompute on union
qi = {r: i for i, r in enumerate(q["q_row"].to_list())}; si = {r: i for i, r in enumerate(s1["s1_row"].to_list())}
QA = va.transform(q["addr_core"].fill_null("").to_list()); SA = va.transform(s1["addr_core"].fill_null("").to_list())
QN = vn.transform(q["sk"].to_list()); SN = vn.transform(s1["sk"].to_list())
ii = np.array([qi[x] for x in d["q_row"].to_list()]); jj = np.array([si[x] for x in d["s1_row"].to_list()])
print("pairs", len(ii), flush=True)
def rowdot(X, Y, ii, jj, B=2_000_000):
    return np.concatenate([np.asarray(X[ii[k:k+B]].multiply(Y[jj[k:k+B]]).sum(1)).ravel() for k in range(0, len(ii), B)])
asim = rowdot(QA, SA, ii, jj); nsim = rowdot(QN, SN, ii, jj)
d = d.with_columns(pl.Series("asim", asim.astype(np.float32)), pl.Series("nsim", nsim.astype(np.float32)))
d = d.with_columns((pl.col("asim") + pl.col("nsim")).alias("comb"), (pl.col("asim") * pl.col("nsim")).alias("prod"))
for c in ("asim","nsim","comb","prod"):
    d = d.with_columns(pl.col(c).rank("ordinal", descending=True).over("q_row").alias("rk_"+c))
print(f"retrieval {time.time()-t0:.0f}s pairs {len(d):,} per q {len(d)/d['q_row'].n_unique():.1f}", flush=True)
d.write_parquet(W+f"v10/addrchan_{split}.parquet")
if split == "train":
    ids = pl.concat([pl.read_parquet(W+f"norm_{split}_s{k}.parquet", columns=["entity_id"]) for k in (2,3)]).with_row_index("q_row").with_columns(pl.col("q_row").cast(pl.UInt32))
    s1id = pl.read_parquet(W+f"norm_{split}_s1.parquet", columns=["entity_id"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    t = gt.join(ids.rename({"entity_id":"s23_id"}), on="s23_id").join(s1id.rename({"entity_id":"s1_id"}), on="s1_id").join(q.select("q_row"), on="q_row", how="semi")
    print("true pairs among eval queries", len(t))
    x = t.join(d, on=["q_row","s1_row"], how="left")
    for c in ("asim","nsim","comb","prod"):
        r = x["rk_"+c].fill_null(999)
        print(c, " ".join("@%d %.3f" % (k, (r <= k).mean()) for k in (1, 3, 10, 30)))
    print("in union", x["asim"].is_not_null().mean())
