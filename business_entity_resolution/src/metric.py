"""Macro F0.5 over Source-1 entities, singletons included (competition metric)."""
import polars as pl


def _f05(tp, n_pred, n_true):
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or tp == 0:
        return 0.0
    p, r = tp / n_pred, tp / n_true
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(s1_ids, pred_pairs: pl.DataFrame, true_pairs: pl.DataFrame, by=None) -> dict:
    """Score predicted (s1_id, s23_id) pairs against true pairs over the given S1 ids.

    by: optional DataFrame (s1_id, group) to also report per-group means.
    """
    ids = pl.DataFrame({"s1_id": list(s1_ids)})
    pred = pred_pairs.select("s1_id", "s23_id").unique().join(ids, on="s1_id", how="semi")
    true = true_pairs.select("s1_id", "s23_id").unique().join(ids, on="s1_id", how="semi")
    tp = pred.join(true, on=["s1_id", "s23_id"]).group_by("s1_id").len("tp")
    npred = pred.group_by("s1_id").len("n_pred")
    ntrue = true.group_by("s1_id").len("n_true")
    df = (
        ids.join(tp, on="s1_id", how="left")
        .join(npred, on="s1_id", how="left")
        .join(ntrue, on="s1_id", how="left")
        .fill_null(0)
    )
    p = pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred")).otherwise(0.0)
    r = pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true")).otherwise(0.0)
    df = df.with_columns(p.alias("p"), r.alias("r")).with_columns(
        pl.when(pl.col("n_true") == 0)
        .then((pl.col("n_pred") == 0).cast(pl.Float64))
        .when(pl.col("tp") == 0)
        .then(0.0)
        .otherwise(1.25 * pl.col("p") * pl.col("r") / (0.25 * pl.col("p") + pl.col("r")))
        .alias("f")
    )
    out = {
        "f05": df["f"].mean(),
        "precision_micro": df["tp"].sum() / max(df["n_pred"].sum(), 1),
        "recall_micro": df["tp"].sum() / max(df["n_true"].sum(), 1),
        "singleton_acc": df.filter(pl.col("n_true") == 0)["f"].mean(),
    }
    if by is not None:
        g = df.join(by, on="s1_id", how="left").group_by(by.columns[1]).agg(pl.col("f").mean())
        out.update({f"f05_{k}": v for k, v in zip(g[by.columns[1]], g["f"])})
    return out


if __name__ == "__main__":
    # Example from the problem statement: expected 0.714; singleton cases 1.0 / 0.0.
    pred = pl.DataFrame({"s1_id": ["a", "a", "a", "c"], "s23_id": ["x", "y", "z", "q"]})
    true = pl.DataFrame({"s1_id": ["a", "a"], "s23_id": ["x", "z"]})
    res = macro_f05(["a"], pred, true)
    assert abs(res["f05"] - 0.7143) < 1e-3, res
    assert macro_f05(["b"], pred, true)["f05"] == 1.0
    assert macro_f05(["c"], pred, true)["f05"] == 0.0
    assert abs(_f05(2, 3, 2) - 0.7143) < 1e-3
    print("metric OK", res)
