"""Are distractors 'orphans' (copies of an S1 absent from the S1 file)? Sibling rate among distractor queries."""
import polars as pl
pl.Config.set_tbl_width_chars(250)
W="../work/"
def q(split):
    d = pl.concat([pl.read_parquet(W+f"norm_{split}_s{k}.parquet", columns=["entity_id","country","name_core","nums","addr_empty"]).with_columns(pl.lit(k).alias("src")) for k in (2,3)]).rename({"entity_id":"s23_id"})
    d = d.with_columns(pl.col("nums").str.split(" ").list.first().fill_null("").alias("n1"))
    # sibling: another query of the OTHER source with same name_core and same first number
    k = d.filter(pl.col("n1")!="").group_by("country","name_core","n1").agg(pl.col("src").n_unique().alias("nsrc"), pl.len().alias("gsz"))
    return d.join(k, on=["country","name_core","n1"], how="left")
tr = q("train").join(pl.read_parquet(W+"gt_pairs.parquet"), on="s23_id", how="left").with_columns(pl.col("s1_id").is_not_null().alias("matched"))
print("TRAIN", tr.group_by("country","matched").agg(pl.len(), (pl.col("nsrc")==2).mean().alias("sib_other_src"), (pl.col("gsz")>1).mean().alias("any_sib")).sort("country","matched"))
te = q("test")
sc = pl.read_parquet(W+"test_scores_lgb_v6k.parquet").sort("p", descending=True).unique("s23_id", keep="first")
te = te.join(sc, on="s23_id", how="left").with_columns(pl.col("p").fill_null(0).cut([0.02, 0.7]).alias("pb"))
print("TEST", te.group_by("country","pb").agg(pl.len(), (pl.col("nsrc")==2).mean().alias("sib_other_src"), (pl.col("gsz")>1).mean().alias("any_sib")).sort("country","pb"))
