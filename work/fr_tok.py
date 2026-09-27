"""Assigned v5 test pairs: how often names differ by a real (non-legal, non-accent) token, per country."""
import polars as pl
from anyascii import anyascii

W = "../work/"
LEGAL = set("sa sas sasu sarl eurl sci snc sc ei eirl scop selarl selas gie inc llc corp co ltd pvt private limited llp "
            "company corporation incorporated pllc lp plc".split())
cols = ["entity_id", "business_name", "business_address", "country", "name_full"]
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=cols)
q = pl.concat([pl.read_parquet(W + f"norm_test_s{k}.parquet", columns=cols) for k in (2, 3)])
sc = pl.read_parquet(W + "test_scores_lgb_v4a.parquet")
m = sc.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= 0.75)
m = (m.join(s1.rename({c: c + "_1" for c in cols if c != "entity_id"}), left_on="s1_id", right_on="entity_id")
      .join(q.drop("country"), left_on="s23_id", right_on="entity_id"))


def toks(e):
    return (e.map_elements(anyascii, return_dtype=pl.String).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", " ")
            .str.split(" ").list.eval(pl.element().filter((pl.element().str.len_chars() >= 3)
                                                         & ~pl.element().is_in(list(LEGAL)))))


m = m.with_columns(toks(pl.col("name_full_1")).alias("t1"), toks(pl.col("name_full")).alias("tq"))
m = m.with_columns(pl.col("tq").list.set_difference("t1").alias("xq"), pl.col("t1").list.set_difference("tq").alias("x1"))
m = m.with_columns((pl.col("xq").list.len() > 0).alias("has_xq"), (pl.col("x1").list.len() > 0).alias("has_x1"))
print(m.group_by("country_1").agg(pl.len(), pl.col("has_xq").mean(), pl.col("has_x1").mean(),
                                  (pl.col("has_xq") & pl.col("has_x1")).mean().alias("swap"),
                                  (pl.col("has_xq") & ~pl.col("has_x1")).mean().alias("add_only"),
                                  (~pl.col("has_xq") & pl.col("has_x1")).mean().alias("drop_only")).sort("country_1"))
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_width_chars(330)
fr = m.filter(pl.col("country_1") == "France")
for kind, f in [("swap", pl.col("has_xq") & pl.col("has_x1")), ("add_only", pl.col("has_xq") & ~pl.col("has_x1"))]:
    x = fr.filter(f)
    print(f"== France {kind}: most common extra query tokens ==")
    print(x.select(pl.col("xq").explode()).to_series().value_counts(sort=True).head(40))
    print(x.sample(30, seed=3).select("p", "business_name_1", "business_name", "business_address_1", "business_address"))
