import sys
sys.path.insert(0, ".")
import polars as pl
W = "../work/"
gt = pl.read_parquet(W + "gt_pairs.parquet")
for split in ("train", "test"):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "country"]).with_columns(pl.lit(k).alias("src")) for k in (2, 3)]).rename({"entity_id": "s23_id"})
    nq = q.group_by("country").len("nq"); n1 = s1.group_by("country").len("n1")
    if split == "train":
        m = gt
    else:
        sc = pl.read_parquet(W + "test_scores_lgb_v5cf.parquet")
        m = sc.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= 0.75).select("s1_id", "s23_id")
    mm = m.join(s1, on="s1_id")
    per = s1.join(mm.group_by("s1_id").len("k"), on="s1_id", how="left").with_columns(pl.col("k").fill_null(0))
    out = per.group_by("country").agg(pl.col("k").mean().alias("mean_k"), (pl.col("k") == 0).mean().alias("single"),
        *[(pl.col("k") == i).mean().alias(f"k{i}") for i in (1, 2, 3, 4, 5, 6)], (pl.col("k") >= 7).mean().alias("k7+"))
    out = out.join(nq, on="country").join(n1, on="country").with_columns((pl.col("nq") / pl.col("n1")).alias("q_per_s1"),
        (pl.col("mean_k") * pl.col("n1") / pl.col("nq")).alias("match_frac"))
    print(split); print(out.sort("country"))
    ms = mm.join(q.select("s23_id", "src"), on="s23_id").group_by("country", "src").len("m").join(q.group_by("country", "src").len("n"), on=["country", "src"]).with_columns((pl.col("m") / pl.col("n")).alias("frac"))
    print(ms.sort("country", "src"))
