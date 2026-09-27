"""Top-1 pairs with equal first house number but p2 < 0.75: what are they (France test vs US/India valid)."""
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(80)
W = "../work/"
src = open(W + "numrel2.py", encoding="utf-8").read().split("one = pl.read_parquet")[0]
exec(src)
def prep(split, f):
    d = pl.read_parquet(W + f)
    d = d.with_columns(pl.col("p2").rank("ordinal", descending=True).over("s23_id").alias("r"),
                       pl.col("p2").sort(descending=True).slice(1, 1).first().over("s23_id").fill_null(0).alias("p2b"))
    d = d.filter(pl.col("r") == 1)
    d = rel(d, split)
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "name_core", "addr_core", "country"]).rename({"entity_id": "s1_id", "name_core": "n1", "addr_core": "a1"})
    s1 = s1.with_columns(pl.when(pl.col("a1").str.len_chars() > 5).then(pl.len().over("country", "a1")).otherwise(1).alias("naddr"))
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "addr_core", "name_full"] if False else ["entity_id", "name_core", "addr_core"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "addr_core": "aq"})
    d = d.join(s1, on="s1_id").join(q, on="s23_id")
    d = d.with_columns(pl.col("nq").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).alias("add"),
                       pl.col("n1").str.split(" ").list.set_difference(pl.col("nq").str.split(" ")).alias("drop"))
    return d.with_columns(pl.concat_str([pl.col("add").list.len().clip(0, 2).cast(pl.String), pl.col("drop").list.len().clip(0, 2).cast(pl.String)], separator="/").alias("ad"),
                          (pl.col("naddr") > 1).alias("sib"), (pl.col("p2b") > 0.05).alias("contest"))
v = prep("train", "valid_scores_stage2.parquet")
t = prep("test", "test_scores_stage2_v9.parquet")
t.write_parquet(W + "top1_test_rel.parquet"); v.write_parquet(W + "top1_val_rel.parquet")
for nm, d, lab in [("VAL", v, True), ("TEST", t, False)]:
    e = d.filter(pl.col("nrel") == "eq")
    ag = [pl.len(), (pl.col("p2") < 0.75).mean().round(3).alias("rej"), pl.col("p2").mean().round(3).alias("p2")]
    if lab: ag += [pl.col("label").mean().round(3).alias("lab")]
    print(nm, "eq by country/ad"); print(e.group_by("country", "ad").agg(ag).filter(pl.col("len") > 2000).sort("country", "ad"))
    print(nm, "eq by country/sib/contest"); print(e.group_by("country", "sib", "contest").agg(ag).sort("country", "sib", "contest"))
fr = t.filter((pl.col("country") == "France") & (pl.col("nrel") == "eq") & (pl.col("p2") < 0.75))
print("FR eq rejected", len(fr))
print(fr.explode("add").group_by("add").len().sort("len", descending=True).head(40))
print(fr.explode("drop").group_by("drop").len().sort("len", descending=True).head(30))
print(fr.filter(~pl.col("contest")).sample(40, seed=1).select("p2", "n1", "nq", "a1", "aq"))
