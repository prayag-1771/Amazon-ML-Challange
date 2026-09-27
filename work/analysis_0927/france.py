"""France vs US/India on test (label-free): uncertainty band share, top-1 score histogram by cell,
what France no-number / empty-address queries look like when v15 leaves them unassigned."""
import polars as pl

pl.Config.set_tbl_rows(50); pl.Config.set_tbl_width_chars(230); pl.Config.set_fmt_str_lengths(70)
W = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/"
sc = pl.read_parquet(W + "test_scores_stage2_v15.parquet")  # cascade pairs: s1_id, s23_id, p, p2
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country", "business_name", "business_address", "name_core"]).rename({"entity_id": "s1_id"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{k}.parquet", columns=["entity_id", "country", "business_name", "business_address", "name_core", "nums", "state", "addr_empty"]) for k in (2, 3)]).rename({"entity_id": "s23_id"})
q = q.with_columns(pl.when(pl.col("addr_empty") == 1).then(pl.lit("empty")).when(pl.col("nums") == "").then(pl.lit("no-number"))
                   .when(pl.col("state") == "").then(pl.lit("no-state")).otherwise(pl.lit("full")).alias("cell"))
top = sc.sort("p2", descending=True).unique("s23_id", keep="first").join(q.select("s23_id", "country", "cell"), on="s23_id")
T = {"US": 0.75, "India": 0.80, "France": 0.75}
bins = [0.01, 0.1, 0.3, 0.5, 0.75, 0.9, 0.99]
print("share of queries with a cascade candidate, and top-1 p2 band shares:")
nq = q.group_by("country").len("nq")
print(top.group_by("country").agg(pl.len().alias("with_cand"), pl.col("p2").is_between(0.05, 0.95).mean().round(4).alias("uncertain_0.05_0.95"),
                                  pl.col("p2").is_between(0.3, 0.9).mean().round(4).alias("uncertain_0.3_0.9"))
      .join(nq, on="country").with_columns((pl.col("with_cand") / pl.col("nq")).round(4).alias("cand_rate")).sort("country"))
h = top.with_columns(pl.col("p2").cut(bins).alias("b")).group_by("country", "b").len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("s"))
print(h.pivot("country", index="b", values="s").sort("b"))
print("cascade-candidate rate by cell:")
print(q.join(top.select("s23_id", pl.lit(1).alias("c")), on="s23_id", how="left").group_by("country", "cell").agg(
    pl.col("c").fill_null(0).mean().round(4).alias("has_cand")).pivot("country", index="cell", values="has_cand").sort("cell"))
print("no-number queries: top-1 p2 bands")
nn = top.filter(pl.col("cell") == "no-number").with_columns(pl.col("p2").cut(bins).alias("b")).group_by("country", "b").len()
print(nn.with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(3).alias("s")).pivot("country", index="b", values="s").sort("b"))
# examples: France no-number, unassigned, highest p2
fr = (top.filter((pl.col("country") == "France") & (pl.col("cell") == "no-number") & (pl.col("p2") < 0.75)).sort("p2", descending=True)
      .join(q.select("s23_id", pl.col("business_name").alias("q_name"), pl.col("business_address").alias("q_addr")), on="s23_id")
      .join(s1.select("s1_id", pl.col("business_name").alias("s1_name"), pl.col("business_address").alias("s1_addr")), on="s1_id"))
print(fr.select("p", "p2", "q_name", "s1_name", "q_addr", "s1_addr").gather_every(max(1, fr.height // 25)).head(25))
