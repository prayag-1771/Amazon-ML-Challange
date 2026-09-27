"""v15 validation loss decomposition with counterfactual gains per error category."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "business_entity_resolution"))
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
W = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/"

s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country", "name_core", "addr_empty"]).rename({"entity_id": "s1_id"})
s1 = s1.with_columns(pl.len().over("country", "name_core").alias("s1_name_cnt"))
s1v = s1.filter(is_valid_expr("s1_id"))
q = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=["entity_id", "country", "name_core", "addr_empty", "nums", "state"]) for k in (2, 3)]).rename({"entity_id": "s23_id"})
gt = pl.read_parquet(W + "gt_pairs.parquet")
gtv = gt.join(s1v.select("s1_id"), on="s1_id", how="semi")
pred = pl.read_parquet(W + "pred15_v.parquet").select("s1_id", "s23_id")
casc = pl.read_parquet(W + "cascade_valid_p001.parquet")
sc = pl.read_parquet(W + "valid_scores_stage2.parquet", columns=["s1_id", "s23_id", "p2"])
ids = s1v["s1_id"]
by = s1v.select("s1_id", "country")

base = macro_f05(ids, pred, gtv, by=by)
print("v15 valid", {k: round(v, 5) for k, v in base.items()})
print("n val S1", len(ids), "true pairs", gtv.height, "pred pairs", pred.join(s1v, on="s1_id", how="semi").height)
ntrue = gtv.group_by("s1_id").len("nt")
print("val S1 by #true matches:", s1v.join(ntrue, on="s1_id", how="left").with_columns(pl.col("nt").fill_null(0).clip(0, 6)).group_by("country", "nt").len().sort("country", "nt").rows())

# ---------------- FN pairs
fn = gtv.join(pred, on=["s1_id", "s23_id"], how="anti")
qn_cnt = s1.group_by("country", "name_core").len("q_name_s1cnt").rename({"name_core": "qname"})
fn = (fn.join(q.rename({"name_core": "qname", "addr_empty": "q_empty"}), on="s23_id", how="left")
        .join(s1.select("s1_id", "s1_name_cnt"), on="s1_id", how="left")
        .join(qn_cnt, on=["country", "qname"], how="left").with_columns(pl.col("q_name_s1cnt").fill_null(0)))
qcasc = casc.select("s23_id").unique().with_columns(pl.lit(True).alias("q_in_casc"))
pcasc = casc.with_columns(pl.lit(True).alias("pair_in_casc"))
qassigned = pred.select("s23_id").unique().with_columns(pl.lit(True).alias("q_assigned"))
fn = (fn.join(qcasc, on="s23_id", how="left").join(pcasc, on=["s1_id", "s23_id"], how="left")
        .join(qassigned, on="s23_id", how="left").join(sc, on=["s1_id", "s23_id"], how="left")
        .with_columns(pl.col("q_in_casc", "pair_in_casc", "q_assigned").fill_null(False)))
cat = (pl.when((pl.col("q_empty") == 1) & ((pl.col("s1_name_cnt") >= 2) | (pl.col("q_name_s1cnt") >= 2))).then(pl.lit("FN empty-addr, name ambiguous"))
       .when(pl.col("q_empty") == 1).then(pl.lit("FN empty-addr, name unique"))
       .when(~pl.col("pair_in_casc")).then(pl.lit("FN addr, blocking/cascade miss"))
       .when(pl.col("q_assigned")).then(pl.lit("FN addr, query given to other S1"))
       .otherwise(pl.lit("FN addr, rejected below threshold")))
fn = fn.with_columns(cat.alias("cat"))

# ---------------- FP pairs
fp = pred.join(s1v.select("s1_id", "country"), on="s1_id").join(gtv, on=["s1_id", "s23_id"], how="anti")
qtrue = gt.select("s23_id").unique().with_columns(pl.lit(True).alias("q_has_true"))
fp = fp.join(qtrue, on="s23_id", how="left").join(ntrue, on="s1_id", how="left").with_columns(
    pl.col("q_has_true").fill_null(False), pl.col("nt").fill_null(0))
fp = fp.with_columns(pl.when(pl.col("q_has_true")).then(pl.lit("FP query belongs to other S1"))
                     .when(pl.col("nt") == 0).then(pl.lit("FP distractor -> singleton S1"))
                     .otherwise(pl.lit("FP distractor -> matched S1")).alias("cat"))

# ---------------- counterfactual gains (fix one category, all else equal)
rows = []
for c in fn["cat"].unique().sort():
    add = fn.filter(pl.col("cat") == c).select("s1_id", "s23_id")
    # a fixed FN query may currently be given to another S1: remove that wrong pair too
    fixed = pred.join(add.select("s23_id"), on="s23_id", how="anti")
    m = macro_f05(ids, pl.concat([fixed, add]), gtv, by=by)
    sub = fn.filter(pl.col("cat") == c)
    rows.append((c, sub.height, sub.filter(pl.col("country") == "US").height, sub.filter(pl.col("country") == "India").height,
                 round(m["f05"] - base["f05"], 5), round(m["f05_US"] - base["f05_US"], 5), round(m["f05_India"] - base["f05_India"], 5)))
for c in fp["cat"].unique().sort():
    rm = fp.filter(pl.col("cat") == c).select("s1_id", "s23_id")
    m = macro_f05(ids, pred.join(rm, on=["s1_id", "s23_id"], how="anti"), gtv, by=by)
    sub = fp.filter(pl.col("cat") == c)
    rows.append((c, sub.height, sub.filter(pl.col("country") == "US").height, sub.filter(pl.col("country") == "India").height,
                 round(m["f05"] - base["f05"], 5), round(m["f05_US"] - base["f05_US"], 5), round(m["f05_India"] - base["f05_India"], 5)))
m = macro_f05(ids, pl.concat([pred.join(fn.select("s23_id"), on="s23_id", how="anti"), fn.select("s1_id", "s23_id")]).join(fp.select("s1_id", "s23_id"), on=["s1_id", "s23_id"], how="anti"), gtv, by=by)
rows.append(("ALL fixed", fn.height + fp.height, 0, 0, round(m["f05"] - base["f05"], 5), 0, 0))
print(pl.DataFrame(rows, schema=["category", "pairs", "US", "India", "gain_F", "gain_US", "gain_IN"], orient="row").sort("gain_F", descending=True))

# ceiling if empty-address ambiguous misses are irreducible
irr = fn.filter(pl.col("cat") == "FN empty-addr, name ambiguous")
rest = fn.filter(pl.col("cat") != "FN empty-addr, name ambiguous")
m = macro_f05(ids, pl.concat([pred.join(rest.select("s23_id"), on="s23_id", how="anti"), rest.select("s1_id", "s23_id")]).join(fp.select("s1_id", "s23_id"), on=["s1_id", "s23_id"], how="anti"), gtv, by=by)
print("ceiling if every error except ambiguous empty-address is fixed:", round(m["f05"], 5), "US", round(m["f05_US"], 5), "IN", round(m["f05_India"], 5))

# FN detail: empty-address ambiguity level, address-present missing components
print(fn.filter(pl.col("q_empty") == 1).with_columns(pl.max_horizontal("s1_name_cnt", "q_name_s1cnt").clip(0, 10).alias("amb"))
      .group_by("country", "amb").len().sort("country", "amb").pivot("country", index="amb", values="len"))
a = fn.filter(pl.col("q_empty") == 0)
print("addr-present FN: share with no house number", round((a["nums"] == "").mean(), 3), "vs all queries", round((q.filter(pl.col("addr_empty") == 0)["nums"] == "").mean(), 3))
fn.write_parquet(W + "lossdec_fn15.parquet"); fp.write_parquet(W + "lossdec_fp15.parquet")
