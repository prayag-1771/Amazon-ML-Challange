"""Empty-address queries: is the true S1 identifiable beyond name_core? + generator structure (matches per S1/source)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "business_entity_resolution"))
import polars as pl
from src.config import is_valid_expr

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
W = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/"
gt = pl.read_parquet(W + "gt_pairs.parquet")
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country", "name_core", "name_full", "legal", "business_name"]).rename({"entity_id": "s1_id"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=["entity_id", "name_core", "name_full", "legal", "business_name", "addr_empty"]).with_columns(pl.lit(k).alias("src")) for k in (2, 3)]).rename({"entity_id": "s23_id"})

# ---- generator structure: matches per S1 per source; empty-address per S1
g = gt.join(q.select("s23_id", "src", "addr_empty"), on="s23_id")
per = g.group_by("s1_id").agg((pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3"), pl.col("addr_empty").sum().alias("ne"), pl.len().alias("n"))
print("matches per S1 per source (n2,n3) top:", per.group_by("n2", "n3").len().sort("len", descending=True).head(12).rows())
print("max n2, n3:", per["n2"].max(), per["n3"].max())
print("empty-address matches per S1:", per.group_by("ne").len().sort("ne").rows())
print("P(S1 has >=1 empty-addr match | n):", per.group_by("n").agg(pl.len(), (pl.col("ne") > 0).mean().round(3)).sort("n").rows())

# ---- empty-address true pairs: how many S1 candidates share name_core / name_full / (name_core, legal)?
e = gt.join(q.filter(pl.col("addr_empty") == 1), on="s23_id").join(s1.select("s1_id", "country"), on="s1_id")
for key_q, key_s, nm in [("name_core", "name_core", "core"), ("name_full", "name_full", "full")]:
    cnt = s1.group_by("country", key_s).len(f"k_{nm}").rename({key_s: "kk"})
    e = e.join(cnt, left_on=["country", key_q], right_on=["country", "kk"], how="left").with_columns(pl.col(f"k_{nm}").fill_null(0))
cnt = s1.group_by("country", "name_core", "legal").len("k_core_legal").rename({"name_core": "kk", "legal": "ll"})
e = e.join(cnt, left_on=["country", "name_core", "legal"], right_on=["country", "kk", "ll"], how="left").with_columns(pl.col("k_core_legal").fill_null(0))
# does the query's own name_core equal its true S1's name_core?
e = e.join(s1.select("s1_id", pl.col("name_core").alias("t_core"), pl.col("name_full").alias("t_full"), pl.col("legal").alias("t_legal")), on="s1_id")
print("empty-addr true pairs:", e.height)
print("query core == true core:", round((e["name_core"] == e["t_core"]).mean(), 3), " full==full:", round((e["name_full"] == e["t_full"]).mean(), 3),
      " legal==legal (both non-empty):", round(e.filter((pl.col("legal") != "") & (pl.col("t_legal") != "")).select((pl.col("legal") == pl.col("t_legal")).mean()).item(), 3))
eq = e.filter(pl.col("name_core") == pl.col("t_core"))
b = lambda c: pl.col(c).clip(0, 5)
print("among core-equal pairs, #S1 sharing core -> #sharing full -> #sharing (core,legal):")
print(eq.group_by(b("k_core").alias("k_core")).agg(pl.len(), (pl.col("k_full") <= 1).mean().round(3).alias("full_unique"),
                                                   (pl.col("k_core_legal") <= 1).mean().round(3).alias("core_legal_unique")).sort("k_core"))
# same-core S1 groups: are their full names identical (true chains) or different?
grp = s1.group_by("country", "name_core").agg(pl.len().alias("k"), pl.col("name_full").n_unique().alias("u_full"), pl.col("legal").n_unique().alias("u_legal")).filter(pl.col("k") >= 2)
print("same-core S1 groups:", grp.height, " share with all-identical full names:", round((grp["u_full"] == 1).mean(), 3))
print("share of S1 in a same-core group, by country:", s1.with_columns(pl.len().over("country", "name_core").alias("k")).group_by("country").agg((pl.col("k") >= 2).mean().round(3)).rows())
# France (test) name uniqueness
for c, sp in (("France", "test"), ("US", "test"), ("India", "test")):
    t = pl.read_parquet(W + f"norm_{sp}_s1.parquet", columns=["country", "name_core"]).filter(pl.col("country") == c)
    print(c, sp, "share of S1 whose name_core is shared:", round(t.with_columns(pl.len().over("name_core").alias("k")).select((pl.col("k") >= 2).mean()).item(), 3))
