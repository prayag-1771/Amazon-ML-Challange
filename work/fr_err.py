"""Label-free precision proxies for the v5 test assignments, per country, plus French samples."""
import polars as pl

W = "../work/"
cols = ["entity_id", "business_name", "business_address", "country", "nums", "addr_empty", "name_full"]
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=cols)
q = pl.concat([pl.read_parquet(W + f"norm_test_s{k}.parquet", columns=cols).with_columns(pl.lit(k).alias("src"))
               for k in (2, 3)])
sc = pl.read_parquet(W + "test_scores_lgb_v4a.parquet")
best = sc.sort("p", descending=True).unique("s23_id", keep="first")
m = best.filter(pl.col("p") >= 0.75)
m = (m.join(s1.rename({c: c + "_1" for c in cols if c != "entity_id"}), left_on="s1_id", right_on="entity_id")
      .join(q, left_on="s23_id", right_on="entity_id"))
m = m.with_columns(
    pl.when((pl.col("nums_1") == "") | (pl.col("nums") == "")).then(None)
      .otherwise(pl.col("nums").str.split(" ").list.contains(pl.col("nums_1").str.split(" ").list.first())).alias("numeq"),
    (pl.col("name_full_1") == pl.col("name_full")).alias("name_eq"),
    pl.col("business_address").str.extract(r"\b(\d{5})\b").alias("pc"),
    pl.col("business_address_1").str.extract(r"\b(\d{5})\b").alias("pc_1"),
)
m = m.with_columns(pl.when(pl.col("pc").is_null() | pl.col("pc_1").is_null()).then(None)
                   .otherwise(pl.col("pc") == pl.col("pc_1")).alias("pceq"))

print("== precision proxies of assigned pairs (p>=0.75) ==")
print(m.group_by("country_1").agg(
    pl.len().alias("n"), pl.col("numeq").is_null().mean().alias("num_missing"),
    (pl.col("numeq") == False).sum().truediv(pl.col("numeq").count()).alias("num_disagree"),
    (pl.col("pceq") == False).sum().truediv(pl.col("pceq").count()).alias("pc_disagree"),
    pl.col("name_eq").mean().alias("name_exact"),
    (pl.col("p") < 0.9).mean().alias("p_lt_0.9"),
).sort("country_1"))

per = m.group_by("s1_id", "country_1", "src").agg(pl.len().alias("k"))
print("== matches per (S1, source) distribution ==")
print(per.group_by("country_1", "src").agg(
    pl.col("k").mean().alias("mean"), (pl.col("k") >= 3).mean().alias("ge3"), (pl.col("k") >= 5).mean().alias("ge5"),
).sort("country_1", "src"))

fr = m.filter(pl.col("country_1") == "France")
show = ["p", "business_name_1", "business_name", "business_address_1", "business_address"]
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(400)
print("== France, confident but house number disagrees ==")
print(fr.filter((pl.col("numeq") == False) & (pl.col("p") > 0.95)).sample(25, seed=1).select(show))
print("== France, confident but postcode disagrees ==")
print(fr.filter((pl.col("pceq") == False) & (pl.col("p") > 0.95)).sample(25, seed=2).select(show))
big = per.filter((pl.col("country_1") == "France") & (pl.col("k") >= 5)).sort("k", descending=True).head(3)
print("== France S1 with most matches from one source ==")
for r in big.iter_rows(named=True):
    print(r)
    print(fr.filter((pl.col("s1_id") == r["s1_id"]) & (pl.col("src") == r["src"])).select(show).head(12))
