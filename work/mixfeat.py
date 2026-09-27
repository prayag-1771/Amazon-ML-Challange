"""Label-free mixture estimate of the true share among equal-house-number pairs, per (country, side, token).
Distractor queries repeat an S1 name with an extra token and a house number shifted up by 1..25; true
matches with the same extra token keep the number. So among top-1 pairs with that token,
true share of 'eq' ~= 1 - r * up / eq, with r = eq/up ratio of pure-distractor tokens (lower quartile)."""
import sys; sys.path.insert(0, "../business_entity_resolution")
import polars as pl
W = "../work/"

def nrel_expr():
    a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False)
    b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
    diff = b - a
    return (pl.when(a.is_null() | b.is_null()).then(pl.lit("miss")).when(diff == 0).then(pl.lit("eq"))
            .when((diff > 0) & (diff <= 25)).then(pl.lit("up25")).when((diff < 0) & (diff >= -25)).then(pl.lit("dn25"))
            .otherwise(pl.lit("far")).alias("nrel"))

def names(split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "name_core", "nums", "country"]).rename(
        {"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename(
        {"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    return s1, q

def pairs(d, s1, q):
    d = d.join(s1, on="s1_id").join(q, on="s23_id")
    t1 = pl.col("n1").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    tq = pl.col("nq").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    return d.with_columns(tq.list.set_difference(t1).alias("add"), t1.list.set_difference(tq).alias("drop"), nrel_expr())

def table(split):
    s1, q = names(split)
    c = pl.read_parquet(W + f"pruned_{split}.parquet", columns=["q_row", "s1_row", "p0_qrank"]).filter(pl.col("p0_qrank") == 1)
    ids1 = s1.select("s1_id")[c["s1_row"].to_numpy()]
    idq = q.select("s23_id")[c["q_row"].to_numpy()]
    d = pairs(pl.concat([ids1, idq], how="horizontal"), s1, q)
    out = []
    for side, other in (("add", "drop"), ("drop", "add")):
        e = d.filter((pl.col(side).list.len() == 1) & (pl.col(other).list.len() == 0)).select("country", "nrel", pl.col(side).list.first().alias("tok"))
        g = e.group_by("country", "tok").agg(pl.len().alias("n"), (pl.col("nrel") == "eq").sum().alias("eq"), (pl.col("nrel") == "up25").sum().alias("up"))
        ref = g.filter((pl.col("n") >= 300) & (pl.col("up") / pl.col("n") > 0.4)).with_columns((pl.col("eq") / pl.col("up")).alias("r"))
        r = ref.group_by("country").agg(pl.col("r").quantile(0.25))
        g = g.join(r, on="country", how="left").with_columns(pl.col("r").fill_null(0.0))
        # smoothing: 20 pseudo-counts toward "noise token" (est 1)
        g = g.with_columns((1 - pl.col("r") * pl.col("up") / (pl.col("eq") + 20 * (1 - pl.col("r")) + 1e-9) ).clip(0, 1).cast(pl.Float32).alias("est"),
                           pl.col("n").log1p().cast(pl.Float32).alias("mlf"))
        print(split, side, r.to_dicts())
        out.append(g.select("country", pl.lit(side).alias("side"), "tok", "est", "mlf"))
    tab = pl.concat(out)
    tab.write_parquet(W + f"mixtab_{split}.parquet")
    return tab

NREL = {"eq": 0, "up25": 1, "dn25": 2, "far": 3, "miss": 4}

def features(d, split):
    """d: s1_id, s23_id (any rows). Returns d with nrel_c, add_n, drop_n, mx_add, mx_drop, mx_addf."""
    s1, q = names(split)
    tab = pl.read_parquet(W + f"mixtab_{split}.parquet")
    x = pairs(d.select("s1_id", "s23_id").unique(), s1, q).with_row_index("i")
    f = x.select("i", "s1_id", "s23_id", pl.col("nrel").replace_strict(NREL).cast(pl.Int8).alias("nrel_c"),
                 pl.col("add").list.len().alias("add_n"), pl.col("drop").list.len().alias("drop_n"))
    for side in ("add", "drop"):
        t = tab.filter(pl.col("side") == side).drop("side")
        e = x.select("i", "country", pl.col(side).alias("tok")).explode("tok").drop_nulls("tok").join(t, on=["country", "tok"], how="left")
        e = e.with_columns(pl.col("est").fill_null(1.0), pl.col("mlf").fill_null(0.0))
        g = e.group_by("i").agg(pl.col("est").min().alias(f"mx_{side}"), pl.col("mlf").max().alias(f"mx_{side}f"))
        f = f.join(g, on="i", how="left").with_columns(pl.col(f"mx_{side}").fill_null(-1.0), pl.col(f"mx_{side}f").fill_null(-1.0))
    return d.join(f.drop("i"), on=["s1_id", "s23_id"], how="left")

if __name__ == "__main__":
    for sp in sys.argv[1:]:
        tab = table(sp)
        for c in tab["country"].unique().sort():
            print(tab.filter((pl.col("country") == c) & (pl.col("side") == "add")).sort("mlf", descending=True).head(15))
