import time, numpy as np, polars as pl, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from src.io_utils import load_ground_truth
gt = load_ground_truth()
for C in ("India","US"):
    s1 = pl.read_parquet("../work/norm_train_s1.parquet").filter(pl.col("country")==C)
    q = pl.concat([pl.read_parquet(f"../work/norm_train_s{i}.parquet") for i in (2,3)]).filter(pl.col("country")==C).sample(30000, seed=1)
    g = gt.join(q.select(pl.col("entity_id").alias("s23_id")), on="s23_id", how="semi")
    t=time.time()
    vn=TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3), min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32).fit(s1["name_core"].to_list())
    va=TfidfVectorizer(analyzer="word", min_df=2, max_df=0.02, sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b").fit((s1["addr_core"]+" "+s1["state"]).to_list())
    N1,Nq=vn.transform(s1["name_core"].to_list()),vn.transform(q["name_core"].to_list())
    A1,Aq=va.transform((s1["addr_core"]+" "+s1["state"]).to_list()),va.transform((q["addr_core"]+" "+q["state"]).to_list())
    print(C,"vec",f"{time.time()-t:.0f}s",flush=True)
    for w in (0.3,0.5,0.7):
        t=time.time()
        X1=sp.hstack([N1*np.sqrt(w),A1*np.sqrt(1-w)]).tocsr(); Xq=sp.hstack([Nq*np.sqrt(w),Aq*np.sqrt(1-w)]).tocsr()
        m=sp_matmul_topn(Xq, X1.T.tocsr(), top_n=30, threshold=0.01, sort=True, n_threads=14).tocoo()
        df=pl.DataFrame({"s23_id":q["entity_id"].to_numpy()[m.row],"s1_id":s1["entity_id"].to_numpy()[m.col],"sim":m.data}).with_columns(pl.col("sim").rank("ordinal",descending=True).over("s23_id").alias("rk"))
        hit=g.join(df,on=["s1_id","s23_id"],how="left")
        print(C,"w",w,f"{time.time()-t:.0f}s"," ".join(f"@{k}:{(hit['rk']<=k).sum()/len(g):.4f}" for k in (1,5,10,20,30)),flush=True)
