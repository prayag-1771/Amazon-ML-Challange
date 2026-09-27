"""Can same-name ambiguity of empty-address queries be broken by (a) legal form / full name, (b) the S1's count of
other (addressed) matches? Train-wide accuracy by group size; then val F gain on top of v15 for unassigned queries."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "business_entity_resolution"))
import numpy as np
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
W = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/"
gt = pl.read_parquet(W + "gt_pairs.parquet")
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country", "name_core", "name_full", "legal"]).rename({"entity_id": "s1_id"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=["entity_id", "country", "name_core", "name_full", "legal", "addr_empty"]) for k in (2, 3)]).rename({"entity_id": "s23_id"})
qa = q.select("s23_id", "addr_empty")
# proxy for "found matches": true matches with a non-empty address (main pipeline recall there is ~99%)
m = gt.join(qa, on="s23_id").filter(pl.col("addr_empty") == 0).group_by("s1_id").len("m")
s1 = s1.join(m, on="s1_id", how="left").with_columns(pl.col("m").fill_null(0))
e = q.filter(pl.col("addr_empty") == 1).drop("addr_empty").join(gt, on="s23_id", how="left").rename({"s1_id": "true_s1"})
# candidates: S1 of the same country with identical name_core
c = e.join(s1.rename({"name_core": "c_core", "name_full": "c_full", "legal": "c_legal"}), left_on=["country", "name_core"], right_on=["country", "c_core"])
c = c.with_columns(pl.len().over("s23_id").alias("k"), (pl.col("legal") == pl.col("c_legal")).alias("leg_eq"),
                   (pl.col("name_full") == pl.col("c_full")).alias("full_eq"), (pl.col("s1_id") == pl.col("true_s1")).fill_null(False).alias("y"))
c = c.with_columns(pl.col("leg_eq").sum().over("s23_id").alias("n_leg"), pl.col("full_eq").sum().over("s23_id").alias("n_full"),
                   (pl.col("m") / pl.col("m").sum().over("s23_id")).alias("m_share"))
print("empty-address queries with >=1 same-core S1:", c["s23_id"].n_unique(), "of", e.height)
# selection rules evaluated per query (k >= 2): pick argmax of a score; accuracy = P(pick is the true S1)
rules = {
    "random": pl.lit(0.0),
    "max m": pl.col("m").cast(pl.Float64),
    "legal eq": pl.col("leg_eq").cast(pl.Float64),
    "full eq": pl.col("full_eq").cast(pl.Float64),
    "full>legal>m": pl.col("full_eq").cast(pl.Float64) * 1000 + pl.col("leg_eq").cast(pl.Float64) * 100 + pl.col("m"),
}
amb = c.filter(pl.col("k") >= 2).with_columns(pl.col("s1_id").hash(7).alias("tie"))
res = []
for nm, sc in rules.items():
    pick = amb.with_columns(sc.alias("sc")).sort(["sc", "tie"], descending=True).unique("s23_id", keep="first")
    res.append(pick.group_by(pl.col("k").clip(2, 6)).agg(pl.col("y").mean().round(3).alias(nm)).sort("k"))
out = res[0]
for r in res[1:]:
    out = out.join(r, on="k")
print(out.join(amb.unique("s23_id").group_by(pl.col("k").clip(2, 6)).len("queries"), on="k"))

# ---- small LightGBM resolver (train on queries touching no val S1, evaluate val)
import lightgbm as lgb
feat = c.with_columns(
    pl.col("m").rank("ordinal", descending=True).over("s23_id").alias("m_rank"),
    (pl.col("m") - pl.col("m").max().over("s23_id")).alias("m_gap"),
    pl.col("leg_eq").cast(pl.Int8), pl.col("full_eq").cast(pl.Int8), (pl.col("c_legal") == "").cast(pl.Int8).alias("c_nolegal"),
    (pl.col("legal") == "").cast(pl.Int8).alias("q_nolegal"))
F = ["k", "m", "m_share", "m_rank", "m_gap", "leg_eq", "full_eq", "n_leg", "n_full", "c_nolegal", "q_nolegal"]
vs1 = s1.filter(is_valid_expr("s1_id")).select("s1_id")
vq = feat.join(vs1, on="s1_id", how="semi").select("s23_id").unique()
tr, va = feat.join(vq, on="s23_id", how="anti"), feat.join(vq, on="s23_id", how="semi")
mdl = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=8),
                lgb.Dataset(tr.select(F).to_numpy().astype(np.float32), tr["y"].to_numpy()), 300)
va = va.with_columns(pl.Series("pr", mdl.predict(va.select(F).to_numpy().astype(np.float32))))
top = va.sort("pr", descending=True).unique("s23_id", keep="first")
print("val resolver top-1 calibration:", top.group_by(pl.col("pr").cut([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])).agg(pl.len(), pl.col("y").mean().round(3)).sort("pr").rows())

# ---- F gain on top of v15 validation predictions: add pairs for empty-address queries v15 left unassigned
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}).filter(is_valid_expr("s1_id"))
gtv = gt.join(s1v, on="s1_id", how="semi")
pred = pl.read_parquet(W + "pred15_v.parquet").select("s1_id", "s23_id")
base = macro_f05(s1v["s1_id"], pred, gtv, by=s1v)
cand = top.join(pred.select("s23_id").unique(), on="s23_id", how="anti")
print(f"v15 {base['f05']:.5f} US {base['f05_US']:.5f} IN {base['f05_India']:.5f}; unassigned val empty-addr queries with same-core S1: {cand.height}")
for th in (0.5, 0.6, 0.7, 0.8, 0.9):
    add = cand.filter(pl.col("pr") >= th).select("s1_id", "s23_id")
    mm = macro_f05(s1v["s1_id"], pl.concat([pred, add]), gtv, by=s1v)
    print(f"  pr>={th}: +{add.height} pairs, precision {cand.filter(pl.col('pr') >= th)['y'].mean():.3f}, F {mm['f05']:.5f} ({mm['f05'] - base['f05']:+.5f}) US {mm['f05_US']:.5f} IN {mm['f05_India']:.5f}")
