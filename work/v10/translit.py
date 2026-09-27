"""Phonetic-skeleton retrieval channel for India addressed queries (transliterated names). Recall probe on train pairs missed by blocking."""
import sys, time; sys.path.insert(0, "../business_entity_resolution")
import numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
W = "../work/"
def skel(col):
    e = pl.col(col).fill_null("").str.to_lowercase()
    for a, b in [(r"[^a-z0-9 ]", ""), ("ph", "f"), ("ck", "k"), (r"c([eiy])", "s$1"), ("c", "k"), ("q", "k"), ("x", "ks"), ("z", "s"), ("w", "v"), ("h", ""),
                 (r"[aeiouy]", ""), ("d", "t"), ("b", "p"), ("g", "k"), ("v", "f"), ("j", "k"), ("m", "n")]:
        e = e.str.replace_all(a, b)
    for ch in "bcdfgklnprstz":
        e = e.str.replace_all(ch + "+", ch)
    return e.str.replace_all(r" +", " ").str.strip_chars()
if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    s1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country","name_core","state"]).with_row_index("s1_row").filter(pl.col("country")=="India")
    q = pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet", columns=["entity_id","country","name_core","state","addr_empty"]) for k in (2,3)]).with_row_index("q_row").filter((pl.col("country")=="India")&(pl.col("addr_empty")==0))
    g = pl.read_parquet(W+"v10/train_gt_rawcov.parquet").filter((pl.col("country")=="India")&(pl.col("addr_empty")==0))
    miss = g.filter(~pl.col("inraw"))
    samp = pl.concat([q.join(miss.select("q_row").unique(), on="q_row", how="semi"), q.sample(100000, seed=0)]).unique("q_row")
    s1 = s1.with_columns(skel("name_core").alias("sk")); samp = samp.with_columns(skel("name_core").alias("sk"))
    print(s1.select("name_core","sk").head(3)); print("queries", len(samp), "missed pairs", len(miss), flush=True)
    parts = []; t0 = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,3), min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32).fit(s1["sk"].to_list())
    for st in s1["state"].unique().to_list():
        a = s1.filter(pl.col("state")==st); b = samp.filter(pl.col("state")==st)
        if len(b) == 0: continue
        m = sp_matmul_topn(vec.transform(b["sk"].to_list()), vec.transform(a["sk"].to_list()).T.tocsr(), top_n=K, threshold=0.05, sort=True, n_threads=14).tocoo()
        parts.append(pl.DataFrame({"q_row": b["q_row"].to_numpy()[m.row], "s1_row": a["s1_row"].to_numpy()[m.col], "sim": m.data}))
    ch = pl.concat(parts).with_columns(pl.col("sim").rank("ordinal", descending=True).over("q_row").alias("rk"))
    print(f"{time.time()-t0:.0f}s", len(ch), flush=True)
    x = miss.join(ch, on=["q_row","s1_row"], how="left")
    for k in (1, 3, 10, K): print("recall@%d %.3f" % (k, (x["rk"] <= k).mean()))
    y = g.filter(pl.col("inraw")).join(samp.select("q_row"), on="q_row", how="semi").join(ch, on=["q_row","s1_row"], how="left")
    print("recall on normal pairs @1 %.3f @10 %.3f" % ((y["rk"] <= 1).mean(), (y["rk"] <= 10).mean()))
    ch.write_parquet(W+"v10/translit_probe.parquet")
