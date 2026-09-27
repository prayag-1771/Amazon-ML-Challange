import polars as pl
W="../work/"
s1=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country","business_name","business_address","name_core"]).rename({"entity_id":"s1_id","name_core":"a"})
q=pl.concat([pl.read_parquet(W+f"norm_test_s{k}.parquet",columns=["entity_id","business_name","business_address","name_core"]) for k in (2,3)]).rename({"entity_id":"s23_id","name_core":"b","business_name":"qn","business_address":"qa"})
sc=pl.read_parquet(W+"test_scores_ens_v8.parquet")
d=sc.join(s1,on="s1_id").join(q,on="s23_id").filter((pl.col("p")>0.02)&(pl.col("country")=="France"))
d=d.with_columns(pl.col("a").str.split(" ").list.unique().alias("ta"),pl.col("b").str.split(" ").list.unique().alias("tb"))
d=d.with_columns(pl.col("tb").list.set_difference("ta").alias("add"),pl.col("ta").list.set_difference("tb").alias("rem"))
qs=set()
for w in ["groupe","developpement","comite","sportive"]:
    s=d.filter((pl.col("add").list.len()==1)&(pl.col("add").list.first()==w)&(pl.col("rem").list.len()==0))
    print("=====",w,len(s),"acc",(s["p"]>=0.7).mean())
    for r in s.sample(6,seed=2).iter_rows(named=True):
        print(f"### Q: {r['qn']} | {r['qa']}")
        allc=sc.filter(pl.col("s23_id")==r["s23_id"]).join(s1,on="s1_id").sort("p",descending=True)
        for c in allc.iter_rows(named=True): print(f"    p={c['p']:.3f} {c['business_name']} | {c['business_address']}")
