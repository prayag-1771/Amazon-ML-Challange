import time, numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking import _topk
from src.io_utils import load_ground_truth
s1 = pl.read_parquet("../work/norm_train_s1.parquet").filter(pl.col("country")=="India")
q = pl.read_parquet("../work/norm_train_s2.parquet").filter(pl.col("country")=="India").sample(50000, seed=1)
gt = load_ground_truth()
for name, fn, kw in [
  ("char34", lambda d: d["name_core"], dict(analyzer="char_wb", ngram_range=(3,4), min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32)),
  ("char24", lambda d: d["name_core"], dict(analyzer="char_wb", ngram_range=(2,4), min_df=2, max_df=0.3, sublinear_tf=True, dtype=np.float32)),
  ("word_na", lambda d: d["name_core"]+" "+d["addr_core"]+" "+d["state"], dict(analyzer="word", ngram_range=(1,1), min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b")),
  ("char34_na", lambda d: d["name_core"]+" "+d["addr_core"], dict(analyzer="char_wb", ngram_range=(3,4), min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32)),
]:
    t=time.time(); vec=TfidfVectorizer(**kw).fit(fn(s1).to_list()); X1=vec.transform(fn(s1).to_list()); Xq=vec.transform(fn(q).to_list()); t1=time.time()-t
    t=time.time(); r,c,v=_topk(Xq,X1,20,chunk=50000); t2=time.time()-t
    pairs=pl.DataFrame({"s23_id":q["entity_id"].to_numpy()[r],"s1_id":s1["entity_id"].to_numpy()[c]})
    g=gt.join(q.select(pl.col("entity_id").alias("s23_id")),on="s23_id",how="semi")
    rec=g.join(pairs,on=["s1_id","s23_id"],how="semi").height/g.height
    print(f"{name}: vec {t1:.0f}s topk {t2:.1f}s recall@20 {rec:.4f} nfeat {X1.shape[1]}", flush=True)
