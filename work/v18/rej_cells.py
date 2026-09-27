"""Are some stage-2 rejections (addressed records, top-1 p2 below the country cutoff) mostly true matches?

Cells as in v17/fr_cells.py (name / house number / legal form / street) x p2 band. Cells are SELECTED on one half of
the validation S1 (hash) and MEASURED on the other half, both directions, so the selection cannot fit noise.
Validation records here are all records with a cascade candidate on a validation S1, so false merges are visible.
A cell is worth accepting only above ~0.75 precision (a false merge costs ~3x a true match).

  python v18/rej_cells.py      (from work/)
"""
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parent / "business_entity_resolution"))
sys.path.insert(0, str(WORK / "v17"))
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05
from fr_cells import CELL, cells

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
W = str(WORK) + "/"
T = {"US": 0.75, "India": 0.80}
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}).filter(is_valid_expr("s1_id"))
s1v = s1v.with_columns((pl.col("s1_id").hash(11) % 2).alias("half"))
gt = pl.read_parquet(W + "gt_pairs.parquet").join(s1v, on="s1_id", how="semi")
pred = pl.read_parquet(W + "pred15_v.parquet").select("s1_id", "s23_id")
v = pl.read_parquet(W + "valid_scores_stage2.parquet", columns=["s1_id", "s23_id", "p2", "label"]).join(s1v, on="s1_id")
top = v.sort("p2", descending=True).unique("s23_id", keep="first").with_columns(pl.col("country").replace_strict(T).alias("th"))
rej = top.filter((pl.col("p2") < pl.col("th")) & (pl.col("p2") >= 0.01)).join(pred, on="s23_id", how="anti")
rej = cells(rej, "train").with_columns(pl.col("p2").cut([0.1, 0.3, 0.5]).cast(pl.Utf8).alias("band"))
K = CELL + ["band", "country"]
print(f"rejected top-1 pairs with p2 in [0.01, cutoff): {rej.height:,}, true-match rate {rej['label'].mean():.3f}")
base = macro_f05(s1v["s1_id"], pred, gt, by=s1v)
for sel_half in (0, 1):
    ev_half = 1 - sel_half
    st = rej.filter(pl.col("half") == sel_half).group_by(K).agg(pl.len().alias("n"), pl.col("label").mean().alias("prec"))
    good = st.filter((pl.col("n") >= 30) & (pl.col("prec") >= 0.85)).select(K)
    ev = rej.filter(pl.col("half") == ev_half).join(good, on=K, how="semi")
    ids = s1v.filter(pl.col("half") == ev_half)
    g = gt.join(ids, on="s1_id", how="semi")
    p = pred.join(ids, on="s1_id", how="semi")
    m0 = macro_f05(ids["s1_id"], p, g)["f05"]
    m1 = macro_f05(ids["s1_id"], pl.concat([p, ev.select("s1_id", "s23_id")]), g)["f05"]
    print(f"select on half {sel_half}: {good.height} cells; on half {ev_half}: +{ev.height} pairs, precision {ev['label'].mean() if ev.height else float('nan'):.3f}, "
          f"F {m1 - m0:+.5f} (on that half)")
    if good.height:
        print(st.join(good, on=K, how="semi").sort("n", descending=True).head(12))
