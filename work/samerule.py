"""Rule test: exact addr_core + no S1 sibling at that address -> lower threshold. Validation + test impact."""
import sys, time
sys.path.insert(0, ".")
import polars as pl
from src.model import load_model, predict
from src.metric import macro_f05
from src.config import is_valid_expr
W = "../work/"
model, feats, T = load_model("lgb_v5cf")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats)))
va.select("s1_id", "s23_id", "p", "label").write_parquet(W + "valid_scores_lgb_v5cf.parquet")
gt = pl.read_parquet(W + "gt_pairs.parquet")

def prep(split, sc):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "name_full", "addr_core", "country"]).rename({"entity_id": "s1_id", "name_full": "n1", "addr_core": "a1"})
    sib = s1.filter(pl.col("a1") != "").group_by("country", "a1").len("a1_n")
    s1 = s1.join(sib, on=["country", "a1"], how="left")
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "name_full", "addr_core"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "name_full": "nq", "addr_core": "aq"})
    top = sc.sort("p", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
    top = top.with_columns(((pl.col("a1") != "") & (pl.col("a1") == pl.col("aq"))).alias("same"),
                           pl.col("nq").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).list.len().alias("nxq"),
                           pl.col("n1").str.split(" ").list.set_difference(pl.col("nq").str.split(" ")).list.len().alias("nx1"))
    return top

top = prep("train", va.select("s1_id", "s23_id", "p"))
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
def score(sel):
    m = macro_f05(s1v["s1_id"], sel, gt, by=s1v)
    return round(m["f05"], 5), round(m["f05_US"], 5), round(m["f05_India"], 5)
print("base", T, score(top.filter(pl.col("p") >= T)))
for sibmax in (1, 2):
    for tl in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        cond = pl.col("same") & (pl.col("a1_n") <= sibmax)
        sel = top.filter((pl.col("p") >= T) | (cond & (pl.col("p") >= tl)))
        print("sib<=", sibmax, "tl", tl, score(sel), "added", sel.height - top.filter(pl.col("p") >= T).height)
# true rate in the added region on valid
lab = top.join(gt.with_columns(pl.lit(1).alias("y")), on=["s1_id", "s23_id"], how="left").with_columns(pl.col("y").fill_null(0))
print(lab.filter(pl.col("same") & (pl.col("a1_n") == 1) & (pl.col("p") < T)).with_columns(pl.col("p").cut([0.02, 0.1, 0.3, 0.5]).alias("b"))
      .group_by("country", "b").agg(pl.len(), pl.col("y").mean()).sort("country", "b"))
# test impact
ts = pl.read_parquet(W + "test_scores_lgb_v5cf.parquet")
tt = prep("test", ts)
for tl in (0.05, 0.1, 0.3):
    print("test added tl", tl, tt.filter(pl.col("same") & (pl.col("a1_n") == 1) & (pl.col("p") >= tl) & (pl.col("p") < T)).group_by("country").len().sort("country").rows())
