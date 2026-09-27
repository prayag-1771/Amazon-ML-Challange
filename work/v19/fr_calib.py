"""v19 diagnostic: is stage 2 systematically under-confident on France?

Anchor cells = (name, house number, legal form, street) cells from v17/fr_cells.py that are >= 99.9% true matches on
US/India validation (n >= 2000). If the data generator is shared across countries, those cells are ~pure in France too,
so any extra rejection there is stage-2 miscalibration, not decoys. Decoy cells (purity <= 2%) show the other side:
how much a lower France cutoff would let in. Label-free on France; no leaderboard feedback used.

Result (2026-09-27): France rejects 0.71% of anchor-cell pairs vs 0.016% on validation, but these cells use the crude
street-overlap test, where "rue" makes different streets look equal. With the fuzzy street-name test of v17/fr_rule.py
the equal-number excess is only 2.3x and validation says those rejections are right. So there is NO evidence of a global
France under-confidence that a lower France cutoff would fix. No cell has <= 2% purity with n >= 2000.

  python v19/fr_calib.py      (from work/)   -> v19/fr_calib_val_top1.parquet (cached), stdout report
"""
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parent / "business_entity_resolution"))
sys.path.insert(0, str(WORK / "v17"))
import polars as pl
from src.config import is_valid_expr
from fr_cells import CELL, cells

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(230)
W = str(WORK) + "/"
OUT = WORK / "v19"
cache = OUT / "fr_calib_val_top1.parquet"
if not cache.exists():
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}).filter(is_valid_expr("s1_id"))
    v = pl.read_parquet(W + "valid_scores_stage2.parquet", columns=["s1_id", "s23_id", "p2", "label"]).join(s1v, on="s1_id")
    cells(v.sort("p2", descending=True).unique("s23_id", keep="first"), "train").select(*CELL, "p2", "label", "country").write_parquet(cache)
vt = pl.read_parquet(cache)
ft = pl.read_parquet(WORK / "v17" / "fr_top1.parquet").select(*CELL, "p2")
st = vt.group_by(CELL).agg(pl.len().alias("n_val"), pl.col("label").mean().alias("purity"))
pure = st.filter((pl.col("purity") >= 0.999) & (pl.col("n_val") >= 2000)).select(CELL)
decoy = st.filter((pl.col("purity") <= 0.02) & (pl.col("n_val") >= 2000)).select(CELL)
TS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
rate = lambda d, t: (d["p2"] < t).mean()
for name, sel in (("PURE anchor cells (val purity >= 0.999)", pure), ("DECOY cells (val purity <= 0.02)", decoy)):
    v, f = vt.join(sel, on=CELL, how="semi"), ft.join(sel, on=CELL, how="semi")
    print(f"\n{name}: validation {v.height:,} pairs, France {f.height:,} pairs")
    print("  share with p2 < t  (validation | France):")
    for t in TS:
        print(f"    t={t:<5} {rate(v, t):.5f} | {rate(f, t):.5f}")
    print("  p2 quantiles 1/5/10/25%  validation:", [round(v["p2"].quantile(q), 4) for q in (0.01, 0.05, 0.1, 0.25)],
          " France:", [round(f["p2"].quantile(q), 4) for q in (0.01, 0.05, 0.1, 0.25)])
# per pure cell: rejection at 0.75
per = (vt.join(pure, on=CELL, how="semi").group_by(CELL).agg(pl.len().alias("n_val"), (pl.col("p2") < 0.75).mean().alias("val_rej"))
       .join(ft.join(pure, on=CELL, how="semi").group_by(CELL).agg(pl.len().alias("n_fr"), (pl.col("p2") < 0.75).mean().alias("fr_rej"),
             (pl.col("p2") < 0.3).mean().alias("fr_rej_03")), on=CELL, how="left").sort("n_fr", descending=True))
print("\nper anchor cell, share rejected at 0.75:")
print(per.with_columns(pl.col("val_rej", "fr_rej", "fr_rej_03").round(5)))
