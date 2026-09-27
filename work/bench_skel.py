import time, numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from src.io_utils import load_ground_truth
def skel(e):
    e = e.str.replace_all("ph","f").str.replace_all("ch","k").str.replace_all("sh","s").str.replace_all("th","t").str.replace_all("bh","b").str.replace_all("kh","k").str.replace_all("gh","g").str.replace_all("dh","d")
    e = e.str.replace_all("[cq]","k").str.replace_all("[vw]","v").str.replace_all("z","j").str.replace_all("x","ks")
    e = e.str.replace_all(r"\B[aeiouy]+","")   # drop non-initial vowels
    for ch in "bcdfghjklmnprstv":
        e = e.str.replace_all(ch + "+", ch)
    return e.str.replace_all(" ", "")
gt = load_ground_truth()
C="India"
s1 = pl.read_parquet("../work/norm_train_s1.parquet").filter(pl.col("country")==C)
q = pl.concat([pl.read_parquet(f"../work/norm_train_s{i}.parquet") for i in (2,3)]).filter(pl.col("country")==C).sample(30000, seed=1)
g = gt.join(q.select(pl.col("entity_id").alias("s23_id")), on="s23_id", how="semi")
fn = lambda d: d.select(skel(pl.col("name_core"))).to_series()
print(list(zip(fn(q.head(8)).to_list(), q.head(8)["name_core"].to_list())))
for ng in [(2,3),(3,4)]:
    t=time.time(); vec=TfidfVectorizer(analyzer="char", ngram_range=ng, min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32).fit(fn(s1).to_list())
    X1=vec.transform(fn(s1).to_list()); Xq=vec.transform(fn(q).to_list())
    m=sp_matmul_topn(Xq, X1.T.tocsr(), top_n=30, threshold=0.01, sort=True, n_threads=14).tocoo()
    df=pl.DataFrame({"s23_id":q["entity_id"].to_numpy()[m.row],"s1_id":s1["entity_id"].to_numpy()[m.col],"sim":m.data}).with_columns(pl.col("sim").rank("ordinal",descending=True).over("s23_id").alias("rk"))
    hit=g.join(df,on=["s1_id","s23_id"],how="left")
    print(ng, f"{time.time()-t:.0f}s", " ".join(f"@{k}:{(hit['rk']<=k).sum()/len(g):.4f}" for k in (1,5,10,30)), flush=True)
