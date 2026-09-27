"""name= & num~ top-1 pairs: shift direction x street overlap x S1 same-name count. US/India truth vs France acceptance + up/down symmetry."""
import sys; sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(220)
W = "../work/"
def addov(x, split):
    s1 = pl.read_parquet(W+f"norm_{split}_s1.parquet", columns=["entity_id","addr_core","alpha_comps"]).rename({"entity_id":"s1_id","addr_core":"a1","alpha_comps":"c1"})
    q = pl.concat([pl.read_parquet(W+f"norm_{split}_s{i}.parquet", columns=["entity_id","addr_core","alpha_comps"]) for i in (2,3)]).rename({"entity_id":"s23_id","addr_core":"aq","alpha_comps":"cq"})
    x = x.join(s1, on="s1_id").join(q, on="s23_id")
    tok = lambda a, c: pl.col(a).fill_null("").str.split(" ").list.set_difference(pl.col(c).fill_null("").str.replace_all(r"\|", " ").str.split(" ")).list.eval(pl.element().filter(~pl.element().str.contains(r"^\d") & (pl.element() != "")))
    x = x.with_columns(tok("a1","c1").alias("s1t"), tok("aq","cq").alias("sqt"))
    x = x.with_columns((pl.col("s1t").list.set_intersection("sqt").list.len() / pl.max_horizontal(pl.col("s1t").list.len(), pl.col("sqt").list.len(), 1)).alias("ov"))
    s1n = pl.read_parquet(W+f"norm_{split}_s1.parquet", columns=["country","name_core"]).group_by("country","name_core").len("ncnt").rename({"name_core":"n1"})
    x = x.join(s1n, on=["country","n1"], how="left")
    return x.with_columns(pl.when(pl.col("ov") > .67).then(pl.lit("st=")).when(pl.col("ov") > .01).then(pl.lit("st~")).otherwise(pl.lit("st!")).alias("ovb"),
                          pl.when(pl.col("ncnt") == 1).then(pl.lit("uniq")).otherwise(pl.lit("dup")).alias("nun"),
                          pl.col("rel").str.extract(r"^(down|up|transp|alpha)").alias("dir"))
t = pl.read_parquet(W+"v10/fr_numdiff.parquet").filter(pl.col("nm2") == "name=")
from importlib import util
spec = util.spec_from_file_location("frnum_rel", W+"v10/frnum.py")
v = pl.read_parquet(W+"v10/ui_cat_v12.parquet").filter((pl.col("nu") == "num~") & (pl.col("nm") == "name="))
f1 = pl.col("u1").fill_null("").str.split(" ").list.first(); fq = pl.col("uq").fill_null("").str.split(" ").list.first()
d = fq.str.extract(r"^(\d+)").cast(pl.Int64, strict=False) - f1.str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
v = v.with_columns(pl.when(f1.str.split("").list.sort() == fq.str.split("").list.sort()).then(pl.lit("transp")).when(d.is_null()).then(pl.lit("alpha")).when(d < 0).then(pl.lit("down")).otherwise(pl.lit("up")).alias("rel"))
t = addov(t, "test"); v = addov(v, "train")
K = ["dir","ovb","nun"]
a = v.group_by(K).agg(pl.len().alias("ui_n"), pl.col("label").mean().round(3).alias("ui_true"), pl.col("acc").mean().round(3).alias("ui_acc"))
b = t.group_by(K).agg(pl.len().alias("fr_n"), pl.col("acc").mean().round(3).alias("fr_acc"), pl.col("p").mean().round(3).alias("fr_p"))
print(b.join(a, on=K, how="full", coalesce=True).sort("ovb","nun","dir"))
t.write_parquet(W+"v10/fr_nameeq_numdiff.parquet"); v.write_parquet(W+"v10/ui_nameeq_numdiff.parquet")
