"""Monte-Carlo expected macro F0.5 per country, treating p as calibrated probabilities.
Truth per query: one of its candidates with prob p_i (normalised if sum > 1), else none."""
import sys
sys.path.insert(0, ".")
import numpy as np
import polars as pl
from src.model import load_model, predict
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05

W = "../work/"
model, feats, T = load_model(sys.argv[1])


def mc(df, s1, n=3, seed=0):
    """df: s1_id, s23_id, p (all candidates). s1: s1_id, country."""
    df = df.sort(["s23_id", "p"], descending=[False, True])
    tot = df.group_by("s23_id", maintain_order=True).agg(pl.col("p").sum().alias("sp"))
    df = df.join(tot, on="s23_id").with_columns((pl.col("p") / pl.max_horizontal(pl.col("sp"), pl.lit(1.0))).alias("pn"))
    df = df.with_columns(pl.col("pn").cum_sum().over("s23_id").alias("cum"))
    pred = df.unique("s23_id", keep="first").filter(pl.col("p") >= T).select("s23_id", pl.col("s1_id").alias("s1_pred"))
    q = df.select("s23_id").unique()
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        u = pl.DataFrame({"s23_id": q["s23_id"], "u": rng.random(len(q))})
        d = df.join(u, on="s23_id").filter(pl.col("cum") >= pl.col("u"))
        tru = d.group_by("s23_id").agg(pl.col("s1_id").first().alias("s1_true"))  # first in p-order with cum>=u
        # note: sorted desc by p within query, cum ascending -> first row reaching u
        tp = pred.join(tru, on="s23_id").filter(pl.col("s1_pred") == pl.col("s1_true")).group_by("s1_pred").len("tp").rename({"s1_pred": "s1_id"})
        npd = pred.group_by("s1_pred").len("n_pred").rename({"s1_pred": "s1_id"})
        ntr = tru.group_by("s1_true").len("n_true").rename({"s1_true": "s1_id"})
        g = s1.join(tp, on="s1_id", how="left").join(npd, on="s1_id", how="left").join(ntr, on="s1_id", how="left").fill_null(0)
        P = pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred")).otherwise(0.0)
        R = pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true")).otherwise(0.0)
        g = g.with_columns(P.alias("P"), R.alias("R")).with_columns(
            pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
            .when(pl.col("tp") == 0).then(0.0)
            .otherwise(1.25 * pl.col("P") * pl.col("R") / (0.25 * pl.col("P") + pl.col("R"))).alias("f"))
        out.append(g.group_by("country").agg(pl.col("f").mean(), (pl.col("tp").sum() / pl.col("n_pred").sum()).alias("prec"),
                                            (pl.col("tp").sum() / pl.col("n_true").sum()).alias("rec"),
                                            (pl.col("n_true") == 0).mean().alias("single")).with_columns(pl.lit(k).alias("k")))
    return pl.concat(out).group_by("country").agg(pl.col("f", "prec", "rec", "single").mean()).sort("country")


va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats))).select("s1_id", "s23_id", "p")
s1v = pl.read_parquet(W + "raw_train_s1.parquet").filter(is_valid_expr("entity_id")).select(pl.col("entity_id").alias("s1_id"), "country")
# restrict valid candidates' truth space: candidates may include non-valid S1 (fine, truth can go there)
print("VALID true:", macro_f05(s1v["s1_id"], va.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= T), load_ground_truth(), by=s1v))
print("VALID MC:"); print(mc(va, s1v))
te = pl.read_parquet(W + "test_scores_" + sys.argv[1] + ".parquet")
s1t = pl.read_parquet(W + "raw_test_s1.parquet").select(pl.col("entity_id").alias("s1_id"), "country")
print("TEST MC:"); print(mc(te, s1t))
