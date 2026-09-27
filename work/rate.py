import polars as pl
from src.io_utils import load_source, load_ground_truth
gt=load_ground_truth()
for split in ("train","test"):
    s1=load_source(split,1).select(pl.col("entity_id").alias("s1_id"),"country")
    q=pl.concat([load_source(split,s).select(pl.col("entity_id").alias("s23_id"),"country",pl.col("business_address").is_null().alias("ae")) for s in (2,3)])
    out=s1.group_by("country").len("n_s1").join(q.group_by("country").agg(pl.len().alias("n_q"),pl.col("ae").mean().alias("q_addr_null")),on="country")
    if split=="train":
        m=gt.join(s1,on="s1_id").group_by("country").len("n_match")
        sz=gt.join(s1,on="s1_id").group_by("s1_id","country").len().group_by("country").agg((pl.col("len")).mean().alias("per_matched_s1"))
        out=out.join(m,on="country").join(sz,on="country")
    else:
        sc=pl.read_parquet("../work/test_scores_lgb_v6k.parquet")
        top=sc.sort("p",descending=True).unique("s23_id",keep="first").filter(pl.col("p")>=0.70)
        m=top.join(s1,on="s1_id").group_by("country").len("n_match")
        out=out.join(m,on="country")
    out=out.with_columns((pl.col("n_match")/pl.col("n_s1")).alias("m_per_s1"),(pl.col("n_match")/pl.col("n_q")).alias("q_matched"),(pl.col("n_q")/pl.col("n_s1")).alias("q_per_s1"))
    print(split);print(out)
