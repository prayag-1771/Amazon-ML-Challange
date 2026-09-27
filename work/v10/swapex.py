import polars as pl
W="../work/"
s1=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country","business_name","business_address","name_core"]).rename({"entity_id":"s1_id","name_core":"a"}).filter(pl.col("country")=="France")
q=pl.concat([pl.read_parquet(W+f"norm_test_s{k}.parquet",columns=["entity_id","business_name","business_address","name_core"]) for k in (2,3)]).rename({"entity_id":"s23_id","name_core":"b","business_name":"qn","business_address":"qa"})
d=pl.read_parquet(W+"test_scores_ens_v8.parquet").join(s1,on="s1_id").join(q,on="s23_id").filter(pl.col("p")>0.02)
d=d.with_columns(pl.col("a").str.split(" ").list.unique().alias("ta"),pl.col("b").str.split(" ").list.unique().alias("tb"))
d=d.with_columns(pl.col("ta").list.set_difference("tb").alias("da"),pl.col("tb").list.set_difference("ta").alias("db"))
for w in ["fils","services","france","amicale","club"]:
    s=d.filter(((pl.col("db").list.len()==1)&(pl.col("db").list.first()==w))|((pl.col("da").list.len()==1)&(pl.col("da").list.first()==w)))
    print("=====",w,len(s),"acc",(s["p"]>=0.7).mean(), "da_len", s["da"].list.len().value_counts().sort("da").rows(), "db_len", s["db"].list.len().value_counts().sort("db").rows())
    for r in s.sample(8,seed=1).iter_rows(named=True):
        print(f"  p={r['p']:.3f} S1: {r['business_name']} | {r['business_address']}\n          Q: {r['qn']} | {r['qa']}")
