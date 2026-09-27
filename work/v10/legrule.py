import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
exec(open("../work/v10/street.py").read().split("v = st(")[0])
W = "../work/"
def rule(split, sc):
    x = st(split, sc, False)
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "legal"]).rename({"entity_id": "s1_id", "legal": "l1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "legal"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "legal": "lq"})
    x = x.join(s1, on="s1_id").join(q, on="s23_id")
    return x.filter((pl.col("l1").fill_null("") == "") & (pl.col("lq").fill_null("") != "") & (pl.col("ov") > .34) & (pl.col("u1") == pl.col("uq")))
T = {"US": .75, "India": .8, "France": .75}
v = rule("train", pl.read_parquet(W + "valid_scores_stage2.parquet")).filter(is_valid_expr("s1_id"))
for fl in (.5, .8, .9, .95):
    r = v.filter((pl.col("p") >= fl) & (pl.col("p2") < pl.col("country").replace_strict(T)))
    print(f"valid p>={fl}: newly accepted {r.height}, precision {r['label'].mean()}", r.group_by("country").agg(pl.len(), pl.col("label").sum()).rows())
t = rule("test", pl.read_parquet(W + "test_scores_stage2_v11.parquet").join(pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id")).filter(pl.col("country") == "France")
for fl in (.5, .8, .9, .95):
    r = t.filter((pl.col("p") >= fl) & (pl.col("p2") < .75))
    print(f"France p>={fl}: newly accepted {r.height}, S1s {r['s1_id'].n_unique()}", r.group_by("lq").len().sort("len", descending=True).rows()[:6])
t.write_parquet(W + "v10/legrule_fr.parquet")
