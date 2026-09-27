"""House-number relation (equal / small jitter / other / missing) for single-added-token top-1 pairs:
train by label, test France by token."""
import polars as pl
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(250)
W = "../work/"
def rel(d, split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "nums"]).rename({"entity_id": "s1_id", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "nums": "uq"})
    d = d.join(s1, on="s1_id").join(q, on="s23_id")
    a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False)
    b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
    diff = (b - a)
    return d.with_columns(pl.when(a.is_null() | b.is_null()).then(pl.lit("miss"))
        .when(diff == 0).then(pl.lit("eq")).when((diff > 0) & (diff <= 25)).then(pl.lit("up25"))
        .when((diff < 0) & (diff >= -25)).then(pl.lit("dn25")).otherwise(pl.lit("far")).alias("nrel"))
one = pl.read_parquet(W + "trtok_one.parquet")
one = rel(one, "train")
cls = pl.when(pl.col("label") == 1).then(pl.lit("pos")).otherwise(pl.lit("neg"))
print(one.with_columns(cls.alias("cls")).group_by("country", "cls", "nrel").len().with_columns(
    (pl.col("len") / pl.col("len").sum().over("country", "cls")).round(3).alias("sh")).sort("country", "cls", "nrel"))
t = pl.read_parquet(W + "test_scores_stage2_v9.parquet").sort("p2", descending=True).unique("s23_id", keep="first")
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "name_core", "country"]).rename({"entity_id": "s1_id", "name_core": "n1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id", "name_core"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq"})
t = t.join(s1, on="s1_id").join(q, on="s23_id")
t = t.with_columns(pl.col("nq").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).alias("add"),
                   pl.col("n1").str.split(" ").list.set_difference(pl.col("nq").str.split(" ")).alias("drop"))
t = t.filter((pl.col("add").list.len() == 1) & (pl.col("drop").list.len() == 0)).with_columns(pl.col("add").list.first().alias("tok"))
t = rel(t, "test")
t.write_parquet(W + "tetok_one.parquet")
for c in ("US", "India", "France"):
    x = t.filter(pl.col("country") == c)
    top = x.group_by("tok").len().sort("len", descending=True).head(25)["tok"]
    x = x.filter(pl.col("tok").is_in(top.implode()))
    pv = x.group_by("tok", "nrel").len().pivot(on="nrel", index="tok", values="len").fill_null(0)
    cols = [k for k in ["eq", "up25", "dn25", "far", "miss"] if k in pv.columns]
    pv = pv.with_columns(pl.sum_horizontal(cols).alias("n")).with_columns([(pl.col(k) / pl.col("n")).round(3) for k in cols])
    pm = x.group_by("tok").agg(pl.col("p2").mean().round(3).alias("p2"))
    print(c); print(pv.join(pm, on="tok").sort("n", descending=True))
