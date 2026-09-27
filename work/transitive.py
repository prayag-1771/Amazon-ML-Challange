"""How much can record-to-record linking recover? For each validation true pair, does another S2/S3 record with the
same normalised raw name (or same name_core+addr_core) exist that is confidently assigned (p>=0.9) to the TRUE S1,
and how often does that key point to a different S1 (precision of the rule)?"""
import sys
sys.path.insert(0, ".")
import polars as pl
from src.model import load_model, predict
from src.config import is_valid_expr
W = "../work/"
model, feats, T = load_model("lgb_v5cf")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats)))
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = pl.read_parquet(W + "gt_pairs.parquet").join(s1, on="s1_id", how="semi")
cols = ["entity_id", "business_name", "name_core", "addr_core", "addr_empty"]
q = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=cols) for k in (2, 3)]).rename({"entity_id": "s23_id"})
q = q.with_columns(pl.col("business_name").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]", "").alias("raw"))
top = va.sort("p", descending=True).unique("s23_id", keep="first")
conf = top.filter(pl.col("p") >= 0.9).select("s23_id", "s1_id")
conf = conf.join(q.select("s23_id", "raw", "name_core"), on="s23_id")
# queries of interest: all queries touching validation S1 (from candidates) + true queries of validation S1
allq = pl.concat([va.select("s23_id"), gt.select("s23_id")]).unique().join(q, on="s23_id")
for key in ["raw", "name_core"]:
    # link: query -> S1s of confident records sharing key (excluding itself)
    link = allq.select("s23_id", key).join(conf.rename({"s23_id": "o_id"}), on=key).filter(pl.col("o_id") != pl.col("s23_id"))
    link = link.group_by("s23_id").agg(pl.col("s1_id").unique().alias("l_s1"), pl.len().alias("l_n"))
    link = link.filter(pl.col("l_s1").list.len() == 1).with_columns(pl.col("l_s1").list.first().alias("l_s1"))
    d = link.join(top.select("s23_id", pl.col("s1_id").alias("cur"), pl.col("p").alias("cur_p")), on="s23_id", how="left")
    d = d.join(gt.rename({"s1_id": "true"}), on="s23_id", how="left")
    d = d.with_columns((pl.col("cur_p").fill_null(0) >= T).alias("assigned"))
    new = d.filter(~pl.col("assigned"))
    print(key, "unassigned queries with a unique link:", len(new),
          "link correct:", (new["l_s1"] == new["true"]).sum(), "link wrong (true other S1):", ((new["true"].is_not_null()) & (new["l_s1"] != new["true"])).sum(),
          "true = none:", new["true"].is_null().sum())
    for n in (1, 2, 3):
        x = new.filter(pl.col("l_n") >= n)
        print("   l_n>=", n, len(x), "correct", (x["l_s1"] == x["true"]).sum(), "none", x["true"].is_null().sum())
    x = new.join(q.select("s23_id", "addr_empty"), on="s23_id")
    for e in (0, 1):
        y = x.filter(pl.col("addr_empty") == e)
        print("   addr_empty", e, len(y), "correct", (y["l_s1"] == y["true"]).sum(), "none", y["true"].is_null().sum())
