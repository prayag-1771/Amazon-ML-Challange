"""Verify the pasted notes against v15 validation: empty-address loss split (no candidate vs found-below-cutoff),
rescue coverage, rare-name winnable part, and the 'fewer addressed matches' signal."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "business_entity_resolution"))
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
W = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}).filter(is_valid_expr("s1_id"))
gt = pl.read_parquet(W + "gt_pairs.parquet")
gtv = gt.join(s1v, on="s1_id", how="semi")
pred = pl.read_parquet(W + "pred15_v.parquet")
p0 = pred.select("s1_id", "s23_id")
base = macro_f05(s1v["s1_id"], p0, gtv, by=s1v)
fn = pl.read_parquet(W + "lossdec_fn15.parquet")


def gain(add):
    fixed = p0.join(add.select("s23_id"), on="s23_id", how="anti")
    m = macro_f05(s1v["s1_id"], pl.concat([fixed, add.select("s1_id", "s23_id")]), gtv, by=s1v)
    return round(m["f05"] - base["f05"], 5), round(m["f05_US"] - base["f05_US"], 5), round(m["f05_India"] - base["f05_India"], 5)


# ---- 1. empty-address FN split: query has no cascade candidate vs pair in cascade but rejected
e = fn.filter(pl.col("q_empty") == 1).with_columns(
    pl.when(~pl.col("q_in_casc")).then(pl.lit("A no cascade candidate"))
    .when(pl.col("pair_in_casc")).then(pl.lit("B found, below cutoff"))
    .otherwise(pl.lit("C other candidates only")).alias("grp"))
print("1. empty-address FN pairs by group / country, with fix-all gain (F, US, IN):")
for g in e["grp"].unique().sort():
    for c in ("US", "India"):
        x = e.filter((pl.col("grp") == g) & (pl.col("country") == c))
        print(f"   {g:28s} {c:6s} pairs {x.height:6d}  gain {gain(x)}")
    print(f"   {g:28s} both   pairs {e.filter(pl.col('grp') == g).height:6d}  gain {gain(e.filter(pl.col('grp') == g))}")
print("   all empty-address FN gain", gain(e))

# ---- 2. rescue coverage: val true pairs with empty-address query absent from cascade
casc = pl.read_parquet(W + "cascade_valid_p001.parquet").select("s23_id").unique()
qe = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=["entity_id", "addr_empty"]) for k in (2, 3)]).rename({"entity_id": "s23_id"}).filter(pl.col("addr_empty") == 1)
tp = gtv.join(qe, on="s23_id", how="semi").join(casc, on="s23_id", how="anti").join(s1v, on="s1_id")
tp = tp.join(pred.filter(pl.col("ch") == "rescue").select("s1_id", "s23_id", pl.lit(1).alias("hit")), on=["s1_id", "s23_id"], how="left")
print("2. empty-address true pairs with no cascade candidate, rescue hit rate:",
      tp.group_by("country").agg(pl.len(), pl.col("hit").fill_null(0).mean().round(3)).sort("country").rows())

# ---- 3. rare-name winnable part in group A: true S1 name_core unique in country
a = e.filter(pl.col("grp") == "A no cascade candidate")
s1n = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "name_core"]).rename({"entity_id": "s1_id", "name_core": "t_core"})
a = a.join(s1n, on="s1_id").with_columns(pl.col("qname").str.split(" ").alias("wq"), pl.col("t_core").str.split(" ").alias("wt"))
a = a.with_columns(pl.when(pl.col("qname") == pl.col("t_core")).then(pl.lit("same core"))
                   .when(pl.col("wt").list.set_difference("wq").list.len() == 0).then(pl.lit("query adds words"))
                   .when(pl.col("wq").list.set_difference("wt").list.len() == 0).then(pl.lit("query drops words"))
                   .otherwise(pl.lit("typo/other")).alias("rel"))
print("3. group A by true-name rarity (s1_name_cnt) and name relation:")
print(a.with_columns(pl.col("s1_name_cnt").clip(1, 11).alias("t_cnt")).group_by("country", pl.when(pl.col("t_cnt") == 1).then(pl.lit("rare(1)"))
      .when(pl.col("t_cnt") <= 10).then(pl.lit("2-10")).otherwise(pl.lit("11+")).alias("rarity")).len().sort("country", "rarity"))
print(a.filter(pl.col("s1_name_cnt") == 1).group_by("country", "rel").len().sort("country", "len", descending=[False, True]))
print("   rare-name group A gain:", gain(a.filter(pl.col("s1_name_cnt") == 1)))

# ---- 4. found-below-cutoff US: p2 distribution + ambiguity
b = e.filter((pl.col("grp") == "B found, below cutoff") & (pl.col("country") == "US"))
print("4. US found-below-cutoff: p2 bands", b.group_by(pl.col("p2").cut([0.01, 0.1, 0.3, 0.5, 0.75])).len().sort("p2").rows())
print("   ambiguity (max of true-name / query-name S1 count):", b.group_by(pl.max_horizontal("s1_name_cnt", "q_name_s1cnt").clip(0, 11).alias("amb")).len().sort("amb").rows())

# ---- 5. 'fewer addressed matches' signal: in same-core tie groups, pick the S1 with MIN addressed matches
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country", "name_core", "legal"]).rename({"entity_id": "s1_id"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=["entity_id", "country", "name_core", "legal", "addr_empty"]) for k in (2, 3)]).rename({"entity_id": "s23_id"})
m = gt.join(q.select("s23_id", "addr_empty"), on="s23_id").filter(pl.col("addr_empty") == 0).group_by("s1_id").len("m")
s1 = s1.join(m, on="s1_id", how="left").with_columns(pl.col("m").fill_null(0))
ee = q.filter(pl.col("addr_empty") == 1).drop("addr_empty").join(gt, on="s23_id").rename({"s1_id": "true_s1"})
c = ee.join(s1.rename({"legal": "c_legal"}), on=["country", "name_core"]).with_columns(pl.len().over("s23_id").alias("k"),
    (pl.col("s1_id") == pl.col("true_s1")).alias("y"), (pl.col("legal") == pl.col("c_legal")).cast(pl.Float64).alias("leg"),
    pl.col("s1_id").hash(7).alias("tie")).filter(pl.col("k") >= 2)
res = None
for nm, sc in {"random": pl.lit(0.0), "max m": pl.col("m").cast(pl.Float64), "min m": -pl.col("m").cast(pl.Float64),
               "legal": pl.col("leg"), "legal, then min m": pl.col("leg") * 100 - pl.col("m")}.items():
    r = c.with_columns(sc.alias("sc")).sort(["sc", "tie"], descending=True).unique("s23_id", keep="first").group_by(pl.col("k").clip(2, 6)).agg(pl.col("y").mean().round(3).alias(nm))
    res = r if res is None else res.join(r, on="k")
print("5. accuracy of picking the owner among same-core S1 (k = group size):")
print(res.join(c.unique("s23_id").group_by(pl.col("k").clip(2, 6)).len("queries"), on="k").sort("k"))
print("   P(S1 has an empty-address match) by m:", s1.join(ee.select(pl.col("true_s1").alias("s1_id")).unique().with_columns(pl.lit(1).alias("h")), on="s1_id", how="left")
      .group_by(pl.col("m").clip(0, 8)).agg(pl.len(), pl.col("h").fill_null(0).mean().round(3)).sort("m").rows())
