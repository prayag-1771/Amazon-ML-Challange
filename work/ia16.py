import sys; sys.path.insert(0,'.')
import polars as pl
from src import india_addr as ia
W=ia.W
cand=pl.concat([pl.read_parquet(W+"test_scores_stage2_v15.parquet",columns=["s1_id","s23_id"]),pl.read_parquet(W+"rescue_test_accept.parquet").select("s1_id","s23_id")])
d=ia.score(ia.pairs("test",cand)); d.write_parquet(W+"india_addr_test_pairs_v15c.parquet")
top=d.sort("pr",descending=True).unique("s23_id",keep="first").join(cand.select("s23_id").unique(),on="s23_id",how="anti")
old=pl.read_parquet(W+"india_addr_test_accept.parquet")
t9=top.filter(pl.col("pr")>=.9)
print("t9",t9.height,"old",old.height,"overlap",t9.join(old,on=["s1_id","s23_id"],how="semi").height)
for th in (.6,.7,.8,.9): print(th,top.filter(pl.col("pr")>=th).height)
