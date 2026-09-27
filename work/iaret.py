import sys; sys.path.insert(0,'../business_entity_resolution')
import numpy as np, polars as pl, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from src.india_addr import skel
exec(open("inmiss2.py").read().split("print(nr.group_by")[0])
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country","name_core","state","addr_core"]).filter(pl.col("country")=="India").with_columns(skel("name_core").alias("sk"),pl.col("addr_core").fill_null(""))
Q=nr.select("s23_id","s1_id").join(q.select("s23_id","nq","aq","stq"),on="s23_id").with_columns(skel("nq").alias("sk"),pl.col("aq").fill_null(""))
va=TfidfVectorizer(analyzer="word",token_pattern=r"\S+",sublinear_tf=True,dtype=np.float32).fit(s1["addr_core"].to_list())
vn=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,3),min_df=2,sublinear_tf=True,dtype=np.float32).fit(s1["sk"].to_list())
QA,SA=va.transform(Q["aq"].to_list()),va.transform(s1["addr_core"].to_list())
QN,SN=vn.transform(Q["sk"].to_list()),vn.transform(s1["sk"].to_list())
QC,SC=sp.hstack([QA,QN]).tocsr(),sp.hstack([SA,SN]).tocsr()
qst,sst=Q["stq"].to_numpy(),s1["state"].to_numpy(); sid=s1["entity_id"].to_numpy(); tru=Q["s1_id"].to_numpy()
for nm,(X,Y) in {"addr":(QA,SA),"name":(QN,SN),"comb":(QC,SC)}.items():
  for K in (30,100):
    hit=np.zeros(len(Q),bool)
    for st in np.unique(qst[qst!=None]).tolist():
        bi,ai=np.flatnonzero(qst==st),np.flatnonzero(sst==st)
        if len(ai)==0: continue
        m=sp_matmul_topn(X[bi],Y[ai].T.tocsr(),top_n=K,threshold=0.02,sort=True,n_threads=4).tocoo()
        g=bi[m.row]; hit[g[sid[ai[m.col]]==tru[g]]]=True
    print(nm,K,hit.sum(),len(Q))
