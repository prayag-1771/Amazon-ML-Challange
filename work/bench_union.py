import time, numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from src.io_utils import load_ground_truth
gt = load_ground_truth(); C="India"
s1 = pl.read_parquet("../work/raw_train_s1.parquet").filter(pl.col("country")==C).select("entity_id").join(pl.read_parquet("../work/norm_train_s1.parquet"), on="entity_id")
q = pl.concat([pl.read_parquet(f"../work/raw_train_s{i}.parquet") for i in (2,3)]).filter(pl.col("country")==C).sample(30000, seed=1).select("entity_id").join(pl.concat([pl.read_parquet(f"../work/norm_train_s{i}.parquet") for i in (2,3)]), on="entity_id")
g = gt.join(q.select(pl.col("entity_id").alias("s23_id")), on="s23_id", how="semi")
fn = lambda d: d["name_core"]+" "+d["addr_core"]+" "+d["state"]
vec=TfidfVectorizer(analyzer="word", min_df=2, max_df=0.02, sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b").fit(fn(s1).to_list())
m=sp_matmul_topn(vec.transform(fn(q).to_list()), vec.transform(fn(s1).to_list()).T.tocsr(), top_n=30, threshold=0.01, sort=True, n_threads=14).tocoo()
tf=pl.DataFrame({"s23_id":q["entity_id"].to_numpy()[m.row],"s1_id":s1["entity_id"].to_numpy()[m.col],"sim":m.data}).with_columns(pl.col("sim").rank("ordinal",descending=True).over("s23_id").alias("rk"))
e5=pl.read_parquet("../work/bench_emb_india.parquet")
for ke,kt in [(10,5),(10,10),(20,10),(20,20),(30,20),(30,30)]:
    u=pl.concat([e5.filter(pl.col("rk")<=ke).select("s23_id","s1_id"), tf.filter(pl.col("rk")<=kt).select("s23_id","s1_id")]).unique()
    print(ke,kt,f"recall {g.join(u,on=['s1_id','s23_id'],how='semi').height/len(g):.4f} avg {len(u)/len(q):.1f}")
u=pl.concat([e5.select("s23_id","s1_id"), tf.select("s23_id","s1_id")]).unique()
miss=g.join(u,on=["s1_id","s23_id"],how="anti").join(s1.select(pl.col("entity_id").alias("s1_id"),pl.col("business_name").alias("n1"),pl.col("business_address").alias("a1")),on="s1_id").join(q.select(pl.col("entity_id").alias("s23_id"),pl.col("business_name").alias("n2"),pl.col("business_address").alias("a2")),on="s23_id")
with open("../work/miss2.txt","w",encoding="utf-8") as f:
    for r in miss.head(40).iter_rows(named=True): f.write(f"{r['n1']} | {r['n2']}\n    {r['a1']}\n    {r['a2']}\n")
