import sys, time; sys.path.insert(0,".")
import numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from src.blocking import _frames
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
K=int(sys.argv[1]) if len(sys.argv)>1 else 5
s1,q=_frames("train")
qe=q.filter(pl.col("addr_core").str.len_chars()==0)
parts=[]
for c in sorted(s1["country"].unique().to_list()):
    a=s1.filter(pl.col("country")==c); b=qe.filter(pl.col("country")==c)
    t=time.time()
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,3),min_df=2,sublinear_tf=True,dtype=np.float32)
    X1=vec.fit_transform(a["name_core"].to_list()).T.tocsr(); Xq=vec.transform(b["name_core"].to_list())
    for i in range(0,Xq.shape[0],200_000):
        m=sp_matmul_topn(Xq[i:i+200_000],X1,top_n=K,threshold=0.3,sort=True,n_threads=14).tocoo()
        parts.append(pl.DataFrame({"q_row":b["q_row"].to_numpy()[m.row+i].astype(np.uint32),"s1_row":a["s1_row"].to_numpy()[m.col].astype(np.uint32),"nm_sim":m.data.astype(np.float32)}))
    print(c,len(a),len(b),f"{time.time()-t:.0f}s",flush=True)
nm=pl.concat(parts)
nm.write_parquet(f"../work/cand_name_train_k{K}.parquet")
gt=load_ground_truth()
n1=pl.read_parquet("../work/norm_train_s1.parquet").rename({"entity_id":"s1_id"}).filter(is_valid_expr()).select("s1_id")
s1i=s1.select(pl.col("s1_row").cast(pl.UInt32),pl.col("entity_id").alias("s1_id"),"country")
qi=qe.select(pl.col("q_row").cast(pl.UInt32),pl.col("entity_id").alias("s23_id"))
g=gt.join(n1,on="s1_id").join(s1i,on="s1_id").join(qi,on="s23_id")
pr=pl.read_parquet("../work/pruned_train.parquet").select("q_row","s1_row").with_columns(pl.lit(1).alias("inp"))
g=g.join(pr,on=["q_row","s1_row"],how="left").join(nm.with_columns(pl.lit(1).alias("inn")),on=["q_row","s1_row"],how="left")
print(g.group_by("country").agg(pl.len(),pl.col("inp").is_null().sum().alias("miss_pr"),(pl.col("inp").is_null()&pl.col("inn").is_null()).sum().alias("miss_union"),pl.col("inn").is_null().sum().alias("miss_name")))
new=nm.join(pr,on=["q_row","s1_row"],how="anti")
print("pairs name:",len(nm),"new vs pruned:",len(new),"empty q:",len(qe),"new/emptyq:",len(new)/len(qe))
