import polars as pl
from src.io_utils import load_source, load_ground_truth
def show(split, country, n, seed, scores):
    sc=pl.read_parquet(scores)
    s1=load_source(split,1).select(pl.col("entity_id").alias("s1_id"),pl.col("business_name").alias("n1"),pl.col("business_address").alias("a1"),"country")
    q=pl.concat([load_source(split,s) for s in (2,3)]).select(pl.col("entity_id").alias("s23_id"),pl.col("business_name").alias("nq"),pl.col("business_address").alias("aq"))
    top=sc.sort("p",descending=True).unique("s23_id",keep="first").join(q,on="s23_id")
    if split=="train":
        gt=load_ground_truth().with_columns(pl.lit(1).alias("y"))
        top=top.join(gt,on=["s1_id","s23_id"],how="left").with_columns(pl.col("y").fill_null(0))
    else: top=top.with_columns(pl.lit(-1).alias("y"))
    ids=s1.filter(pl.col("country")==country).join(top.select("s1_id").unique(),on="s1_id",how="semi").sample(n,seed=seed)
    for r in ids.iter_rows(named=True):
        print(f"S1: {r['n1']} | {r['a1']}")
        for x in top.filter(pl.col("s1_id")==r["s1_id"]).sort("p",descending=True).iter_rows(named=True):
            print(f"   p={x['p']:.3f} y={x['y']}  {x['nq']} | {x['aq']}")
import sys
if sys.argv[1]=="val":
    # valid scores contain s1_id? build from valid_scores
    show("train","US",12,5,"../work/valid_scores_lgb_v6k.parquet")
else:
    show("test","France",14,7,"../work/test_scores_lgb_v6k.parquet")
