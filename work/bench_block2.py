import time, numpy as np, polars as pl, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from src.io_utils import load_ground_truth
gt = load_ground_truth()
C = "India"
s1 = pl.read_parquet("../work/norm_train_s1.parquet").filter(pl.col("country")==C)
q = pl.concat([pl.read_parquet(f"../work/norm_train_s{i}.parquet") for i in (2,3)]).filter(pl.col("country")==C).sample(30000, seed=1)
g = gt.join(q.select(pl.col("entity_id").alias("s23_id")), on="s23_id", how="semi")
def evaluate(name, fn, kw, k=20):
    t=time.time(); vec=TfidfVectorizer(**kw).fit(fn(s1).to_list()); X1=vec.transform(fn(s1).to_list()); Xq=vec.transform(fn(q).to_list()); t1=time.time()-t
    t=time.time(); m=sp_matmul_topn(Xq, X1.T.tocsr(), top_n=k, threshold=0.01, sort=True, n_threads=14).tocoo(); t2=time.time()-t
    pairs=pl.DataFrame({"s23_id":q["entity_id"].to_numpy()[m.row],"s1_id":s1["entity_id"].to_numpy()[m.col], "rank": np.zeros(len(m.row))})
    h = g.join(pairs, on=["s1_id","s23_id"], how="semi").height/g.height
    print(f"{name}: vec {t1:.0f}s topk {t2:.1f}s ({t2/len(q)*1e3:.2f} ms/q) recall@{k} {h:.4f} nfeat {X1.shape[1]}", flush=True)
    return pairs
W = dict(analyzer="word", min_df=2, max_df=0.02, sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b")
p1 = evaluate("word_addr", lambda d: d["addr_core"]+" "+d["state"], W)
p2 = evaluate("word_name_addr", lambda d: d["name_core"]+" "+d["addr_core"], W)
p3 = evaluate("char3_name", lambda d: d["name_core"], dict(analyzer="char_wb", ngram_range=(3,3), min_df=2, max_df=0.02, sublinear_tf=True, dtype=np.float32))
u = pl.concat([p1,p2,p3]).unique(["s1_id","s23_id"])
print("union recall", g.join(u, on=["s1_id","s23_id"], how="semi").height/g.height, "avg cands", len(u)/len(q))
p4 = evaluate("char34_name_addr", lambda d: d["name_core"]+" "+d["addr_core"], dict(analyzer="char_wb", ngram_range=(3,4), min_df=2, max_df=0.02, sublinear_tf=True, dtype=np.float32))
u = pl.concat([p1,p2,p3,p4]).unique(["s1_id","s23_id"])
print("union4 recall", g.join(u, on=["s1_id","s23_id"], how="semi").height/g.height, "avg cands", len(u)/len(q))
miss = g.join(u, on=["s1_id","s23_id"], how="anti")
m = miss.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("name_core").alias("n1"), pl.col("addr_core").alias("a1")), on="s1_id").join(q.select(pl.col("entity_id").alias("s23_id"), pl.col("business_name").alias("raw2"), pl.col("name_core").alias("n2"), pl.col("addr_core").alias("a2")), on="s23_id")
with open("../work/miss.txt","w",encoding="utf-8") as f:
    for r in m.head(60).iter_rows(named=True):
        f.write(f"{r['n1']!r} | {r['n2']!r} [{r['raw2']!r}]\n     {r['a1'][:70]!r}\n     {r['a2'][:70]!r}\n")
