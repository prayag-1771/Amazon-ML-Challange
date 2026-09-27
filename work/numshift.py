import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
W = "../work/"
C = ["s1_id","s23_id","num_first_eq","num_min_rel","q_nums_n","s1_nums_n","q_addr_empty","nc_eq","legal_eq","xq_n","x1_n","q_ncand","state_eq"]
def load(split):
    if split == "val":
        d = pl.read_parquet(W+"valid_feats.parquet", columns=C+["label"]).join(pl.read_parquet(W+"valid_scores_lgb_v6k.parquet").select("s1_id","s23_id","p"), on=["s1_id","s23_id"])
        s1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country"])
    else:
        d = pl.read_parquet(W+"test_feats.parquet", columns=C).join(pl.read_parquet(W+"test_scores_lgb_v6k.parquet"), on=["s1_id","s23_id"]).with_columns(pl.lit(None, pl.Int8).alias("label"))
        s1 = pl.read_parquet(W+"norm_test_s1.parquet", columns=["entity_id","country"])
    s1 = s1.rename({"entity_id":"s1_id"})
    top = d.sort("p", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id")
    n1 = top.select("s1_id","country").unique().group_by("country").len("n1")
    return top, n1
def cat():
    hasnum = (pl.col("q_nums_n")>0)&(pl.col("s1_nums_n")>0)
    return (pl.when(pl.col("q_addr_empty")==1).then(pl.lit("empty")).when(~hasnum).then(pl.lit("nonum"))
            .when(pl.col("num_first_eq")==1).then(pl.lit("num_eq")).otherwise(pl.lit("num_diff"))).alias("cat")
for split in ("val","test"):
    top, n1 = load(split)
    top = top.with_columns(cat(), pl.col("p").cut([0.02,0.3,0.7,0.95,0.999]).alias("b"))
    g = top.group_by("country","cat","b").agg(pl.len().alias("n"), pl.col("label").mean().alias("y")).join(n1, on="country").with_columns((pl.col("n")/pl.col("n1")).round(4).alias("per_s1"))
    print(split); print(g.filter(pl.col("b")!="(-inf, 0.02]").sort("country","cat","b").select("country","cat","b","per_s1","y"))
