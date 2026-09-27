"""Bias-corrected validation gain of rescue v2, with the correction split by record type.

Validation records are (a) records whose true S1 is a validation S1 and (b) unmatched records with any validation-S1
candidate. Type (a) is ~10% of matched records, so its wrong picks that land on non-validation S1 are mirrored on
test by the other 90% of matched records landing on validation S1: count ALL of its wrong picks. Type (b) is ~100% of
unmatched records (almost all touch a validation S1), so only its wrong picks that land on validation S1 count.

  python v18/eval_split.py A|Bpred       (from work/)
"""
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parent / "business_entity_resolution"))
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05

W = str(WORK) + "/"
fs = sys.argv[1]
va = pl.read_parquet(WORK / "v18" / f"rescue_v2_valid_{fs}.parquet")
gt = pl.read_parquet(W + "gt_pairs.parquet")
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gtv = gt.join(s1v.select("s1_id"), on="s1_id", how="semi")
base = pl.read_parquet(W + "pred15_v.parquet").select("s1_id", "s23_id")
m0 = macro_f05(s1v["s1_id"], base, gtv, by=s1v)["f05"]
kind = pl.concat([gtv.select("s23_id", pl.lit("val_true").alias("kind"))])
top = (va.sort(["pr", "s1_id"], descending=[True, False]).unique("s23_id", keep="first")
       .join(base.select("s23_id").unique(), on="s23_id", how="anti")
       .join(kind, on="s23_id", how="left").with_columns(pl.col("kind").fill_null("unmatched")))
print(f"[{fs}] unassigned validation records with a pick:", top.group_by("kind").len().rows())
rows = []
for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    a = top.filter(pl.col("pr") >= t)
    vis = a.filter(pl.col("vs1"))
    tp = vis.filter(pl.col("label") == 1)
    fp_true_all = a.filter((pl.col("kind") == "val_true") & (pl.col("label") == 0)).height      # all wrong picks of type (a)
    fp_unm_vis = vis.filter(pl.col("kind") == "unmatched").height                                 # visible wrong picks of type (b)
    fp_vis = vis.height - tp.height
    m1 = macro_f05(s1v["s1_id"], pl.concat([base, vis.select("s1_id", "s23_id")]), gtv)["f05"] - m0
    mt = macro_f05(s1v["s1_id"], pl.concat([base, tp.select("s1_id", "s23_id")]), gtv)["f05"] - m0
    rows.append((t, a.height, tp.height, fp_vis, fp_true_all, fp_unm_vis, m1, mt))
e_fp = (rows[0][6] - rows[0][7]) / max(rows[0][3], 1)
print(f"  per false-merge effect {e_fp:.2e}; per true-match effect {rows[0][7] / max(rows[0][2], 1):.2e}")
for t, n, ntp, fpv, fpa, fpu, m1, mt in rows:
    fp = fpa + fpu
    print(f"  t={t}: picks {n:,} TP {ntp:,} | false merges counted {fp:,} (type a all {fpa:,}, type b visible {fpu:,}) "
          f"precision {ntp / max(ntp + fp, 1):.3f} | naive {m1:+.5f} corrected {mt + e_fp * fp:+.5f}")
