"""Label-free blocking-recall proxy: strong rule matches (same name_core + same first house number +
address token overlap) that are NOT in the pruned candidate set. Train (valid S1) gives the rule's precision."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W = "../work/"
def side(split, s):
    d = pl.read_parquet(W + f"norm_{split}_s{s}.parquet", columns=["entity_id", "country", "name_core", "addr_core", "nums", "addr_empty"])
    return d.with_columns(pl.col("nums").str.split(" ").list.first().fill_null("").alias("n1"),
                          pl.col("addr_core").str.replace_all(r"\d+", "").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 2)).list.unique().alias("tok"))
for split in ["train", "test"]:
    s1 = side(split, 1)
    if split == "train":
        s1 = s1.filter(is_valid_expr("entity_id"))
    q = pl.concat([side(split, 2), side(split, 3)])
    k = ["country", "name_core", "n1"]
    j = q.filter((pl.col("name_core") != "") & (pl.col("n1") != "")).join(
        s1.filter((pl.col("name_core") != "") & (pl.col("n1") != "")), on=k, suffix="_1")
    j = j.with_columns((pl.col("tok").list.set_intersection("tok_1").list.len() / pl.col("tok").list.set_union("tok_1").list.len().clip(1)).alias("jac"))
    j = j.filter(pl.col("jac") >= 0.3).select(pl.col("entity_id").alias("s23_id"), pl.col("entity_id_1").alias("s1_id"), "country", "jac")
    # unique rule match per query only
    j = j.filter(pl.len().over("s23_id") == 1)
    cand = pl.read_parquet(W + f"pruned_{split}.parquet", columns=["s1_id", "s23_id"]).with_columns(pl.lit(1).alias("inc"))
    j = j.join(cand, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("inc").fill_null(0))
    if split == "train":
        gt = load_ground_truth().with_columns(pl.lit(1).alias("y"))
        j = j.join(gt, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("y").fill_null(0))
        print(j.group_by("country", "inc").agg(pl.len(), pl.col("y").mean()).sort("country", "inc"))
    else:
        print(j.group_by("country", "inc").agg(pl.len()).sort("country", "inc"))
    n1 = s1.group_by("country").len()
    print(j.filter(pl.col("inc") == 0).group_by("country").len().join(n1, on="country", suffix="_s1").with_columns((pl.col("len") / pl.col("len_s1")).alias("miss_per_s1")))

# ---- where are the test France misses lost? (blocking union vs prune) and what do they look like
from src.blocking import with_ids
fr = j.filter((pl.col("inc") == 0))
u = with_ids(pl.read_parquet(W + "cand_test.parquet", columns=["q_row", "s1_row", "emb_rank", "tf_rank"]), "test")
fr = fr.join(u.select("s1_id", "s23_id", "emb_rank", "tf_rank"), on=["s1_id", "s23_id"], how="left")
print(fr.group_by("country", pl.col("emb_rank").is_not_null().alias("in_union")).len())
from src.io_utils import load_source
a = load_source("test", 1).select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
q = pl.concat([load_source("test", 2), load_source("test", 3)]).select(pl.col("entity_id").alias("s23_id"), pl.col("business_name").alias("nq"), pl.col("business_address").alias("aq"))
x = fr.filter(pl.col("country") == "France").join(a, on="s1_id").join(q, on="s23_id").sample(15, seed=1)
for r in x.iter_rows(named=True):
    print(f"  S1: {r['n1']} | {r['a1']}\n  Q : {r['nq']} | {r['aq']}   union={r['emb_rank'] is not None}")
