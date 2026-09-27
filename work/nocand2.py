import polars as pl
pl.Config.set_tbl_rows(30)
fn=pl.read_parquet("nocand_v.parquet")
print(fn.group_by("country","addr_empty","q_has_cand","in_pruned").len().sort("len",descending=True))
r=pl.read_parquet("v10/rescue_valid.parquet"); print(r.columns)
v=pl.read_parquet("valid_scores_stage2.parquet")
topv=v.sort("p2",descending=True).unique("s23_id",keep="first").select("s23_id",pl.col("p2").alias("vtop"))
rt=r.sort("pr",descending=True).unique("s23_id",keep="first").join(topv,on="s23_id",how="left")
rt=rt.filter(pl.col("vs1"))
# rescue top for queries that DO have cascade candidates
h=rt.filter(pl.col("vtop").is_not_null())
h=h.with_columns(pl.col("vtop").cut([.1,.3,.5,.75]).alias("vb"),pl.col("pr").cut([.5,.8,.9,.95]).alias("pb"))
print(h.group_by("vb","pb").agg(pl.len(),pl.col("label").mean().round(3)).sort("vb","pb"))
