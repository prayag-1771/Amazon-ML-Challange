"""PU / address-view diagnostic. If exact address agreement is (nearly) independent of name noise given a true match,
then P(same|true)=c per country and P(true|group) ~= same_share(group)/c. Validate on US/India labels, apply to France."""
import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
W = "../work/"
def top1(split, sc):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "name_full", "addr_core", "country"]).rename({"entity_id": "s1_id", "name_full": "n1", "addr_core": "a1"})
    s1 = s1.join(s1.filter(pl.col("a1") != "").group_by("country", "a1").len("a1_n"), on=["country", "a1"], how="left")
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "name_full", "addr_core"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "name_full": "nq", "addr_core": "aq"})
    t = sc.sort("p", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
    t = t.filter((pl.col("a1") != "") & (pl.col("aq") != "")).with_columns(
        ((pl.col("a1") == pl.col("aq")) & (pl.col("a1_n") == 1)).cast(pl.Float64).alias("same"),
        pl.col("nq").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).list.len().alias("nxq"),
        pl.col("n1").str.split(" ").list.set_difference(pl.col("nq").str.split(" ")).list.len().alias("nx1"))
    return t.with_columns(pl.when((pl.col("nxq") == 0) & (pl.col("nx1") == 0)).then(pl.lit("same_tok")).when(pl.col("nx1") == 0).then(pl.lit("added"))
                          .when(pl.col("nxq") == 0).then(pl.lit("dropped")).otherwise(pl.lit("swap")).alias("kind"),
                          pl.col("p").cut([0.02, 0.1, 0.3, 0.5, 0.75, 0.9, 0.98, 0.999]).alias("b"))
va = top1("train", pl.read_parquet(W + "valid_scores_lgb_v5cf.parquet").select("s1_id", "s23_id", "p", "label"))
print("VALID same share | label, kind"); print(va.group_by("country", "label", "kind").agg(pl.len(), pl.col("same").mean().round(4)).sort("country", "label", "kind"))
cv = va.filter(pl.col("p") > 0.999).group_by("country").agg(pl.col("same").mean().alias("c"))
print("c from p>0.999", cv)
g = va.join(cv, on="country").group_by("country", "b").agg(pl.len(), pl.col("label").mean().round(3).alias("true"), (pl.col("same").mean() / pl.col("c").first()).round(3).alias("pi_est")).sort("country", "b")
print(g)
te = top1("test", pl.read_parquet(W + "test_scores_lgb_v5cf.parquet"))
ct = te.filter(pl.col("p") > 0.999).group_by("country").agg(pl.col("same").mean().alias("c"))
print("TEST c", ct)
print("TEST same by kind for p>0.999", te.filter(pl.col("p") > 0.999).group_by("country", "kind").agg(pl.len(), pl.col("same").mean().round(4)).sort("country", "kind"))
g = te.join(ct, on="country").group_by("country", "b").agg(pl.len(), (pl.col("same").mean() / pl.col("c").first()).round(3).alias("pi_est")).sort("country", "b")
print(g)
g = te.join(ct, on="country").group_by("country", "kind", "b").agg(pl.len(), (pl.col("same").mean() / pl.col("c").first()).round(3).alias("pi_est")).sort("country", "kind", "b")
print(g.filter(pl.col("country") == "France"))
# expected true pairs lost / wrongly added per country (label-free estimate)
e = te.join(ct, on="country").with_columns((pl.col("same") / pl.col("c")).alias("w"))
print(e.group_by("country").agg((pl.col("w") * (pl.col("p") < 0.75)).sum().alias("est_true_unassigned"), ((1 - pl.col("w")) * (pl.col("p") >= 0.75)).sum().alias("est_false_assigned"), pl.len()))
