"""S1 entities sharing an exact address: how do their names differ? (tells what the data generator treats as
'different entity'). Also: assigned France word-swap queries whose extra word matches a same-address S1 sibling."""
import polars as pl
from anyascii import anyascii

W = "../work/"
LEGAL = set("sa sas sasu sarl eurl sci snc sc ei eirl scop selarl selas gie inc llc corp co ltd pvt private limited llp "
            "company corporation incorporated pllc lp plc and fils associes".split())


def toks(s):
    s = "".join(c if c.isalnum() else " " for c in anyascii(s or "").lower())
    return frozenset(t for t in s.split() if len(t) >= 3 and t not in LEGAL)


for split in ("train", "test"):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "addr_full", "name_full"])
    s1 = s1.filter(pl.col("addr_full") != "").with_columns(pl.len().over("country", "addr_full").alias("k"))
    g = s1.filter((pl.col("k") >= 2) & (pl.col("k") <= 4))
    pairs = g.join(g, on=["country", "addr_full"]).filter(pl.col("entity_id") < pl.col("entity_id_right"))
    pairs = pairs.sample(min(len(pairs), 200_000), seed=0)
    kinds = []
    for a, b in zip(pairs["name_full"], pairs["name_full_right"]):
        ta, tb = toks(a), toks(b)
        sh = len(ta & tb)
        kinds.append("identical" if ta == tb else ("share_word" if sh else "disjoint"))
    pairs = pairs.with_columns(pl.Series("kind", kinds))
    print(split, pairs.group_by("country", "kind").agg(pl.len()).with_columns(
        (pl.col("len") / pl.col("len").sum().over("country")).round(3).alias("share")).sort("country", "kind"))
    if split == "test":
        pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(50)
        print(pairs.filter((pl.col("country") == "France") & (pl.col("kind") == "share_word"))
                   .sample(25, seed=1).select("name_full", "name_full_right", "addr_full"))
