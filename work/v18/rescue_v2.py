"""v18: empty-address rescue v2. Model training and application on the candidate pairs of work/v10/resc2.py.

The pairs (v10/resc2_pairs_{split}.parquet) come from three name-only retrieval views, per country:
char-3-gram on name_core, char-3-gram on the noise-stripped name, and word TF-IDF on the noise-stripped name.
They cover every empty-address S2/S3 record. This script adds tie features and trains a 3-seed LightGBM.
For each record the pipeline leaves unassigned, it keeps the top-1 pair at or above a cutoff.

Feature sets
  A  resc2 features as designed
  B  A + tie features: s1_m (the S1's number of addressed matches; train: labels, test: base submission),
     its gap to the least-matched S1 of the same name in the query, tie-group size, and whether the legal
     form singles this S1 out of its tie group.

  python v18/rescue_v2.py fit A|B [seeds]        (from work/)  -> v18/rescue_v2_{set}_s{seed}.lgb, v18/rescue_v2_valid_{set}.parquet
  python v18/rescue_v2.py eval A|B
  python v18/rescue_v2.py evalpred B [seeds]        # validation with s1_m from predicted matches (as on test)
  python v18/rescue_v2.py apply A|B <base_dir> <cutoff>   -> v18/rescue_v2_patch.parquet  (US/India only)
"""
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parent / "business_entity_resolution"))
import lightgbm as lgb
import numpy as np
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05

W = str(WORK) + "/"
OUT = WORK / "v18"
F_A = ["s_c", "s_k", "s_w", "rk_c", "rk_k", "rk_w", "g_c", "g_k", "g_w", "n_c", "eq", "eqk", "leq", "is_in", "s1_ncnt", "s1k_ncnt", "q_ncnt",
       "s1_empty", "r", "rk_r", "ts", "jw", "len1", "lenq", "r_gap", "r_ntie", "s1_nq", "fq_in1", "f1_inq", "nq_tok", "n1_tok", "ftok_eq",
       "min_df_miss1", "min_df_shared", "n_miss1", "n_missq", "comb", "rk_comb", "comb_gap", "comb_gap2"]
F_M = ["s1_m", "s1_m_gap", "tie_n", "tie_leq_unique"]
FEATS = {"A": F_A, "B": F_A + F_M}
P = dict(objective="binary", num_leaves=31, min_data_in_leaf=300, learning_rate=0.05, lambda_l2=5, feature_fraction=0.7,
         bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=8)
COUNTRIES = ("US", "India")


def read_pairs(path, col):
    t = pl.read_csv(path, separator="\t", quote_char=None, schema_overrides={col: pl.Utf8}).with_columns(pl.col(col).fill_null(""))
    return t.with_columns(pl.col(col).str.split(",")).explode(col).filter(pl.col(col) != "").select(
        pl.col("source1_entity_id").alias("s1_id"), pl.col(col).alias("s23_id"))


def addressed(split):
    return pl.concat([pl.scan_parquet(W + f"norm_{split}_s{k}.parquet").select(pl.col("entity_id").alias("s23_id"), "addr_empty")
                      .filter(pl.col("addr_empty") == 0).select("s23_id") for k in (2, 3)])


def add_tie_features(d, matches, split):
    """matches: (s1_id, s23_id) pairs used to count each S1's addressed matches."""
    m = matches.lazy().join(addressed(split), on="s23_id", how="semi").group_by("s1_id").len("s1_m").collect()
    d = d.join(m, on="s1_id", how="left").with_columns(pl.col("s1_m").fill_null(0).cast(pl.Int32))
    grp = ["q_row", "n1"]
    return d.with_columns(
        (pl.col("s1_m") - pl.col("s1_m").min().over(grp)).alias("s1_m_gap"),
        pl.len().over(grp).alias("tie_n"),
        ((pl.col("leq") == 1) & ((pl.col("leq") == 1).sum().over(grp) == 1)).cast(pl.Int8).alias("tie_leq_unique"),
    ).drop("n1")


def load(split, matches):
    cols = ["q_row", "s1_id", "s23_id", "country", "n1"] + F_A
    d = pl.read_parquet(W + f"v10/resc2_pairs_{split}.parquet", columns=cols)
    return add_tie_features(d, matches, split)


def fit(fs, seeds):
    t0 = time.time()
    gt = pl.read_parquet(W + "gt_pairs.parquet")
    d = load("train", gt).join(gt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "s23_id"], how="left")
    d = d.with_columns(pl.col("label").fill_null(0), is_valid_expr("s1_id").alias("vs1"))
    # Validation records: those whose TRUE S1 is a validation S1, plus unmatched records with any validation-S1
    # candidate (their false merges count on validation). Training: all other records, minus every row whose
    # candidate S1 is a validation S1 (no validation label is ever trained on). resc2.py held out every record
    # touching a validation S1; with ~25 candidates per record that left only ~7% of the data for training.
    matched = gt.select("s23_id").unique()
    vq = pl.concat([gt.filter(is_valid_expr("s1_id")).select("s23_id"),
                    d.filter(pl.col("vs1")).select("s23_id").join(matched, on="s23_id", how="anti")]).unique()
    tr = d.join(vq, on="s23_id", how="anti").filter(~pl.col("vs1"))
    va = d.join(vq, on="s23_id", how="semi")
    del d
    F = FEATS[fs]
    print(f"[{fs}] train pairs {tr.height:,} (pos {tr['label'].mean():.3f}), valid pairs {va.height:,}, {len(F)} features, {time.time() - t0:.0f}s", flush=True)
    X, y = tr.select(F).to_numpy().astype(np.float32), tr["label"].to_numpy()
    del tr
    Xv = va.select(F).to_numpy().astype(np.float32)
    pr = np.zeros(va.height)
    for s in seeds:
        m = lgb.train({**P, "seed": s}, lgb.Dataset(X, y, feature_name=F), 500)
        m.save_model(str(OUT / f"rescue_v2_{fs}_s{s}.lgb"))
        pr += m.predict(Xv) / len(seeds)
        print(f"  seed {s} done {time.time() - t0:.0f}s", flush=True)
    imp = sorted(zip(F, m.feature_importance("gain")), key=lambda x: -x[1])
    print("  top gain:", [(f, round(g)) for f, g in imp[:12]], flush=True)
    va = va.with_columns(pl.Series("pr", pr))
    va.select("s1_id", "s23_id", "country", "pr", "label", "vs1").write_parquet(OUT / f"rescue_v2_valid_{fs}.parquet")
    evaluate(fs)


def evaluate(fs):
    va = pl.read_parquet(OUT / f"rescue_v2_valid_{fs}.parquet")
    gt = pl.read_parquet(W + "gt_pairs.parquet")
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
    gtv = gt.join(s1v.select("s1_id"), on="s1_id", how="semi")
    base = pl.read_parquet(W + "pred15_v.parquet").select("s1_id", "s23_id")
    topall = (va.sort(["pr", "s1_id"], descending=[True, False]).unique("s23_id", keep="first")
              .join(base.select("s23_id").unique(), on="s23_id", how="anti"))
    top = topall.filter(pl.col("vs1"))
    m0 = macro_f05(s1v["s1_id"], base, gtv, by=s1v)
    print(f"[{fs}] base (v15) F {m0['f05']:.5f} US {m0['f05_US']:.5f} IN {m0['f05_India']:.5f}; unassigned validation records with a pick: "
          f"{topall.height:,} ({top.height:,} pick a validation S1)")
    # Bias correction. Validation records' correct picks all land on validation S1 and are fully counted, but ~90% of
    # their wrong picks land on non-validation S1 and are invisible to the validation metric. On test, the wrong picks
    # of the other 90% of records land on validation S1 at about the same total rate (random 10% S1 split), so the
    # false merges hitting validation S1 ~= ALL wrong picks of validation records. corrected = TP-only gain +
    # (per-false-merge effect measured on the visible ones) x (all wrong picks).
    rows = []
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        a = top.filter(pl.col("pr") >= t)
        n_all = topall.filter(pl.col("pr") >= t).height
        n_tp = int(a["label"].sum())
        n_fp_vis, n_fp_all = a.height - n_tp, n_all - n_tp
        m1 = macro_f05(s1v["s1_id"], pl.concat([base, a.select("s1_id", "s23_id")]), gtv, by=s1v)["f05"] - m0["f05"]
        mt = macro_f05(s1v["s1_id"], pl.concat([base, a.filter(pl.col("label") == 1).select("s1_id", "s23_id")]), gtv, by=s1v)["f05"] - m0["f05"]
        rows.append((t, n_all, n_tp, n_fp_vis, n_fp_all, m1, mt))
    # per-false-merge effect, pooled over the lowest cutoff (most visible false merges)
    e_fp = (rows[0][5] - rows[0][6]) / max(rows[0][3], 1)
    print(f"  per false merge effect on validation F: {e_fp:.2e}")
    for t, n_all, n_tp, n_fp_vis, n_fp_all, m1, mt in rows:
        corr = mt + e_fp * n_fp_all
        print(f"  t={t}: picks {n_all:,}, correct {n_tp:,}, precision (all picks) {n_tp / max(n_all, 1):.3f} | "
              f"naive F {m1:+.5f}  TP-only {mt:+.5f}  corrected {corr:+.5f}", flush=True)


def eval_pred(fs, seeds):
    """Honest check of the tie features: rebuild s1_m for validation S1 from v15's PREDICTED matches (as on test),
    keep labels for the other S1, re-score the validation records with the saved models, evaluate."""
    gt = pl.read_parquet(W + "gt_pairs.parquet")
    pv = pl.read_parquet(W + "pred15_v.parquet").select("s1_id", "s23_id")
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id"]).rename({"entity_id": "s1_id"}).filter(is_valid_expr("s1_id"))
    matches = pl.concat([gt.join(s1v, on="s1_id", how="anti"), pv.join(s1v, on="s1_id", how="semi")])
    old = pl.read_parquet(OUT / f"rescue_v2_valid_{fs}.parquet")
    d = load("train", matches).join(old.select("s1_id", "s23_id", "label", "vs1"), on=["s1_id", "s23_id"], how="semi")
    d = d.join(old.select("s1_id", "s23_id", "label", "vs1"), on=["s1_id", "s23_id"])
    X = d.select(FEATS[fs]).to_numpy().astype(np.float32)
    pr = np.mean([lgb.Booster(model_file=str(OUT / f"rescue_v2_{fs}_s{s}.lgb")).predict(X) for s in seeds], axis=0)
    d.with_columns(pl.Series("pr", pr)).select("s1_id", "s23_id", "country", "pr", "label", "vs1").write_parquet(OUT / f"rescue_v2_valid_{fs}pred.parquet")
    evaluate(fs + "pred")


def apply(fs, base_dir, cutoff, seeds=(1, 2, 3)):
    base = read_pairs(Path(W) / base_dir / "matching_results.tsv", "matched_entity_ids")
    d = load("test", base).filter(pl.col("country").is_in(COUNTRIES))
    F = FEATS[fs]
    X = d.select(F).to_numpy().astype(np.float32)
    pr = np.mean([lgb.Booster(model_file=str(OUT / f"rescue_v2_{fs}_s{s}.lgb")).predict(X) for s in seeds], axis=0)
    d = d.with_columns(pl.Series("pr", pr)).join(base.select("s23_id").unique(), on="s23_id", how="anti")  # records the base leaves unassigned
    top = d.sort(["pr", "s1_id"], descending=[True, False]).unique("s23_id", keep="first")
    print(top.group_by("country").agg(pl.len(), *[(pl.col("pr") >= t).sum().alias(f">={t}") for t in (0.6, 0.7, 0.8, 0.9, 0.95)]).sort("country"))
    patch = top.filter(pl.col("pr") >= cutoff).select("s1_id", "s23_id", "country", "pr")
    patch.write_parquet(OUT / "rescue_v2_patch.parquet")
    print(f"wrote v18/rescue_v2_patch.parquet: {patch.height:,} pairs at pr >= {cutoff}", patch.group_by("country").len().sort("country").rows())


if __name__ == "__main__":
    cmd, fs = sys.argv[1], sys.argv[2]
    if cmd == "fit":
        fit(fs, [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3])
    elif cmd == "eval":
        evaluate(fs)
    elif cmd == "evalpred":
        eval_pred(fs, [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3])
    else:
        apply(fs, sys.argv[3], float(sys.argv[4]))
