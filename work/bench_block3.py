import time, numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from src.io_utils import load_ground_truth
gt = load_ground_truth()
res = {}
for C in ("India", "US"):
    s1 = pl.read_parquet("../work/norm_train_s1.parquet").filter(pl.col("country")==C)
    q = pl.concat([pl.read_parquet(f"../work/norm_train_s{i}.parquet") for i in (2,3)]).filter(pl.col("country")==C).sample(30000, seed=1)
    g = gt.join(q.select(pl.col("entity_id").alias("s23_id")), on="s23_id", how="semi")
    parts = {}
    W = dict(analyzer="word", min_df=2, max_df=0.02, sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b")
    for name, fn, kw in [("na", lambda d: d["name_core"]+" "+d["addr_core"]+" "+d["state"], W),
                         ("a", lambda d: d["addr_core"]+" "+d["state"], W),
                         ("n3", lambda d: d["name_core"], dict(analyzer="char_wb", ngram_range=(3,3), min_df=2, max_df=0.02, sublinear_tf=True, dtype=np.float32))]:
        t=time.time(); vec=TfidfVectorizer(**kw).fit(fn(s1).to_list()); X1=vec.transform(fn(s1).to_list()); Xq=vec.transform(fn(q).to_list())
        m=sp_matmul_topn(Xq, X1.T.tocsr(), top_n=30, threshold=0.01, sort=True, n_threads=14).tocoo()
        df=pl.DataFrame({"s23_id":q["entity_id"].to_numpy()[m.row],"s1_id":s1["entity_id"].to_numpy()[m.col],"sim":m.data})
        df=df.with_columns(pl.col("sim").rank("ordinal",descending=True).over("s23_id").alias("rk"))
        parts[name]=df
        hit=g.join(df,on=["s1_id","s23_id"],how="left")
        print(C, name, f"{time.time()-t:.0f}s", " ".join(f"@{k}:{(hit['rk']<=k).sum()/len(g):.4f}" for k in (1,3,5,10,20,30)), flush=True)
    for ks in [(10,5,5),(15,10,5),(20,10,10),(30,15,10)]:
        u=pl.concat([parts[n].filter(pl.col("rk")<=k).select("s23_id","s1_id") for n,k in zip(("na","a","n3"),ks)]).unique()
        print(C, "union", ks, f"recall {g.join(u,on=['s1_id','s23_id'],how='semi').height/len(g):.4f} avg {len(u)/len(q):.1f}", flush=True)
