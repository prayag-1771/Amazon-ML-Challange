"""Validation F0.5 decomposition: current vs oracle on blocking misses vs oracle on model errors."""
import sys
sys.path.insert(0, ".")
import polars as pl
from src.model import load_model, predict, assign
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05

W = "../work/"
model, feats, T = load_model(sys.argv[1])
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats)))
s1v = pl.read_parquet(W + "raw_train_s1.parquet").filter(is_valid_expr("entity_id")).select(pl.col("entity_id").alias("s1_id"), "country")
gt = load_ground_truth()
gtv = gt.join(s1v, on="s1_id", how="semi")
pred = assign(va, T).select("s1_id", "s23_id")
print("current", macro_f05(s1v["s1_id"], pred, gt, by=s1v))
# pairs of valid S1 whose true pair never appears among candidates
cand = va.select("s1_id", "s23_id").unique()
miss = gtv.join(cand, on=["s1_id", "s23_id"], how="anti")
print("valid true pairs", len(gtv), "blocking misses", len(miss), f"{len(miss)/len(gtv):.4f}")
print("oracle blocking", macro_f05(s1v["s1_id"], pl.concat([pred, miss.select("s1_id", "s23_id")]).unique(), gt, by=s1v))
# oracle model on candidates: predict exactly the true pairs among candidates
inc = gtv.join(cand, on=["s1_id", "s23_id"], how="semi")
print("oracle model (cands only)", macro_f05(s1v["s1_id"], inc, gt, by=s1v))
# miss profile
qa = pl.concat([pl.read_parquet(W + f"norm_train_s{i}.parquet", columns=["entity_id", "addr_empty", "business_name"]) for i in (2, 3)]).rename({"entity_id": "s23_id"})
m = miss.join(qa, on="s23_id").join(s1v, on="s1_id")
print(m.group_by("country").agg(pl.len(), pl.col("addr_empty").mean(), pl.col("business_name").str.contains(r"[^\x00-\x7F]").mean().alias("nonascii")))
# queries with no candidates at all vs candidates-but-wrong-S1
qc = va.select("s23_id").unique()
print("miss with query absent from candidates:", m.join(qc, on="s23_id", how="anti").height, "/", m.height)
