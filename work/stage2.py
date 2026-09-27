"""Stage-2 collective re-ranker: first-stage p + per-query and per-S1 aggregates of first-stage scores.
Trained with S1-grouped CV on the validation queries (first-stage scores there are out-of-sample)."""
import sys
sys.path.insert(0, ".")
import numpy as np, polars as pl, lightgbm as lgb
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
W = "../work/"

def base(split):
    if split == "valid":
        a = pl.read_parquet(W + "valid_scores_lgb_v5cf.parquet").rename({"p": "pa"})
        b = pl.read_parquet(W + "valid_scores_lgb_v6k.parquet").drop("label").rename({"p": "pb"})
        d = a.join(b, on=["s1_id", "s23_id"])
        vf = pl.read_parquet(W + "valid_feats.parquet", columns=["s1_id", "s23_id", "src", "q_addr_empty", "q_nonascii", "s1_name_cnt", "ce_logit", "nc_tset", "ac_tset"])
    else:
        a = pl.read_parquet(W + "test_scores_lgb_v5cf.parquet").rename({"p": "pa"})
        b = pl.read_parquet(W + "test_scores_lgb_v6k.parquet").rename({"p": "pb"})
        d = a.join(b, on=["s1_id", "s23_id"])
        ce = pl.read_parquet(W + "ce_test.parquet").join(
            pl.read_parquet(W + "ce_test_b.parquet").rename({"ce_logit": "ce_b"}), on=["q_row", "s1_row"])
        ce = ce.select("q_row", "s1_row", ((pl.col("ce_logit") + pl.col("ce_b")) / 2).alias("ce_logit"))
        vf = pl.read_parquet(W + "test_feats.parquet", columns=["q_row", "s1_row", "s1_id", "s23_id", "src", "q_addr_empty", "q_nonascii", "s1_name_cnt", "nc_tset", "ac_tset"])
        vf = vf.join(ce, on=["q_row", "s1_row"], how="left").drop("q_row", "s1_row")
    return d.join(vf, on=["s1_id", "s23_id"]).with_columns(((pl.col("pa") + pl.col("pb")) / 2).alias("p"))

def feats(d):
    d = d.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("s23_id").alias("q_rank"),
        (pl.col("p") - pl.col("p").max().over("s23_id")).alias("q_gap_best"),
        pl.col("p").sort(descending=True).slice(1, 1).first().over("s23_id").fill_null(0).alias("q_p2"),
        pl.len().over("s23_id").alias("q_n"),
        pl.len().over("s1_id").alias("s1_n"),
    )
    top = d.filter(pl.col("q_rank") == 1)
    for th, nm in [(0.9, "hi"), (0.5, "mid")]:
        agg = top.filter(pl.col("p") >= th).group_by("s1_id", "src").agg(pl.len().alias("n"))
        a2 = agg.filter(pl.col("src") == 2).select("s1_id", pl.col("n").alias(f"s1_{nm}2"))
        a3 = agg.filter(pl.col("src") == 3).select("s1_id", pl.col("n").alias(f"s1_{nm}3"))
        d = d.join(a2, on="s1_id", how="left").join(a3, on="s1_id", how="left").with_columns(
            pl.col(f"s1_{nm}2").fill_null(0), pl.col(f"s1_{nm}3").fill_null(0))
        own = ((pl.col("q_rank") == 1) & (pl.col("p") >= th)).cast(pl.Int32)
        d = d.with_columns(
            (pl.when(pl.col("src") == 2).then(pl.col(f"s1_{nm}2")).otherwise(pl.col(f"s1_{nm}3")) - own).alias(f"s1_{nm}_same"),
            pl.when(pl.col("src") == 2).then(pl.col(f"s1_{nm}3")).otherwise(pl.col(f"s1_{nm}2")).alias(f"s1_{nm}_other"),
        ).drop(f"s1_{nm}2", f"s1_{nm}3")
    d = d.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("s1_id", "src").alias("s1_src_rank"),
        (pl.col("p") - pl.col("p").max().over("s1_id")).alias("s1_gap_best"),
    )
    # competitor: strongest alternative S1 in this query and its collective support
    d = d.with_columns(
        (pl.col("s1_hi_same") + pl.col("s1_hi_other")).alias("s1_hi_tot"))
    alt = d.select("s23_id", "q_rank", "s1_hi_tot", "s1_hi_same")
    r1 = alt.filter(pl.col("q_rank") == 1).select("s23_id", pl.col("s1_hi_tot").alias("r1_hi_tot"), pl.col("s1_hi_same").alias("r1_hi_same"))
    r2 = alt.filter(pl.col("q_rank") == 2).select("s23_id", pl.col("s1_hi_tot").alias("r2_hi_tot"), pl.col("s1_hi_same").alias("r2_hi_same"))
    d = d.join(r1, on="s23_id", how="left").join(r2, on="s23_id", how="left")
    d = d.with_columns(
        pl.when(pl.col("q_rank") == 1).then(pl.col("r2_hi_tot")).otherwise(pl.col("r1_hi_tot")).fill_null(-1).alias("alt_hi_tot"),
        pl.when(pl.col("q_rank") == 1).then(pl.col("r2_hi_same")).otherwise(pl.col("r1_hi_same")).fill_null(-1).alias("alt_hi_same"),
    ).drop("r1_hi_tot", "r1_hi_same", "r2_hi_tot", "r2_hi_same")
    return d

F = ["p", "pa", "pb", "ce_logit", "nc_tset", "ac_tset", "src", "q_addr_empty", "q_nonascii", "s1_name_cnt",
     "q_rank", "q_gap_best", "q_p2", "q_n", "s1_n", "s1_hi_same", "s1_hi_other", "s1_mid_same", "s1_mid_other",
     "s1_src_rank", "s1_gap_best", "s1_hi_tot", "alt_hi_tot", "alt_hi_same"]
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7, num_threads=14)
R = 400

if __name__ == "__main__":
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
    gt = load_ground_truth()
    d = feats(base("valid"))
    # fold of each query = hash of its top-1 S1 (keeps an S1's queries together)
    qf = d.filter(pl.col("q_rank") == 1).select("s23_id", (pl.col("s1_id").hash(3) % 5).alias("fold"))
    d = d.join(qf, on="s23_id")
    oof = np.zeros(len(d), np.float32)
    for k in range(5):
        tr, te = d["fold"] != k, d["fold"] == k
        m = lgb.train(P, lgb.Dataset(d.filter(tr).select(F).to_numpy(), d.filter(tr)["label"].to_numpy()), R)
        oof[te.to_numpy()] = m.predict(d.filter(te).select(F).to_numpy(), num_threads=14)
    d = d.with_columns(pl.Series("p2", oof))
    top0 = d.sort("p", descending=True).unique("s23_id", keep="first")
    top2 = d.sort("p2", descending=True).unique("s23_id", keep="first")
    for t in [0.6, 0.65, 0.7, 0.75, 0.8]:
        m0 = macro_f05(s1v["s1_id"], top0.filter(pl.col("p") >= t).select("s1_id", "s23_id"), gt, by=s1v)
        m2 = macro_f05(s1v["s1_id"], top2.filter(pl.col("p2") >= t).select("s1_id", "s23_id"), gt, by=s1v)
        print(f"t={t} base {m0['f05']:.5f} (US {m0['f05_US']:.5f} IN {m0['f05_India']:.5f})  stage2 {m2['f05']:.5f} (US {m2['f05_US']:.5f} IN {m2['f05_India']:.5f})", flush=True)
    m = lgb.train(P, lgb.Dataset(d.select(F).to_numpy(), d["label"].to_numpy()), R)
    print(sorted(zip(F, m.feature_importance("gain").round()), key=lambda x: -x[1]))
    m.save_model(W + "stage2.lgb")
    d.select("s1_id", "s23_id", "p", "p2", "label").write_parquet(W + "valid_scores_stage2.parquet")
