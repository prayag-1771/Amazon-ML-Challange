"""Do records of one S1 entity in the same source share an address-format signature (number prefix,
zero padding, state tail, order) more than distractors do? Train truth."""
import sys; sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
W = "../work/"
q = pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet", columns=["entity_id","country","business_address","business_name"]).with_columns(pl.lit(k).alias("src")) for k in (2,3)]).with_row_index("q_row")
a = pl.col("business_address").fill_null("")
q = q.with_columns(
    a.str.extract(r"^\s*([^\d\s]*\s*)\d", 1).fill_null("~").str.strip_chars().alias("pre"),
    a.str.contains(r"(^|\D)0\d").alias("zpad"),
    a.str.split(",").list.last().str.strip_chars().alias("tail"),
    a.str.contains(r"^\s*\D*\d").alias("numfirst"),
    (a.str.to_uppercase()==a).alias("upper"),
    a.str.split(",").list.len().alias("ncomp"),
)
gt = pl.read_parquet(W+"gt_pairs.parquet")
c = pl.read_parquet(W+"pruned_train.parquet", columns=["q_row","s1_id","s23_id","p0_qrank"]).filter(pl.col("p0_qrank")==1)
c = c.join(gt.with_columns(pl.lit(1).alias("y")), on=["s1_id","s23_id"], how="left").with_columns(pl.col("y").fill_null(0))
c = c.join(q.select(pl.col("q_row").cast(pl.UInt32), "country","src","pre","zpad","tail","numfirst","upper","ncomp","business_address","business_name"), on="q_row")
# reference: true records of the same S1, same source (exclude self)
ref = c.filter(pl.col("y")==1).select("s1_id","src",pl.col("s23_id").alias("r_id"),pl.col("pre").alias("r_pre"),pl.col("zpad").alias("r_zpad"),pl.col("tail").alias("r_tail"),pl.col("numfirst").alias("r_nf"),pl.col("ncomp").alias("r_nc"))
x = c.sample(1_500_000, seed=2).join(ref, on=["s1_id","src"]).filter(pl.col("r_id")!=pl.col("s23_id"))
x = x.with_columns((pl.col("pre")==pl.col("r_pre")).alias("m_pre"), (pl.col("zpad")==pl.col("r_zpad")).alias("m_zpad"),
                   (pl.col("tail")==pl.col("r_tail")).alias("m_tail"), (pl.col("numfirst")==pl.col("r_nf")).alias("m_nf"), (pl.col("ncomp")==pl.col("r_nc")).alias("m_nc"))
print(x.group_by("country","src","y").agg(pl.len(), *[pl.col(m).mean() for m in ("m_pre","m_zpad","m_tail","m_nf","m_nc")]).sort("country","src","y"))
# baseline: random pairs of same country/src
print(c.group_by("country","src").agg(pl.col("pre").value_counts(sort=True).head(8)).explode("pre").unnest("pre"))
