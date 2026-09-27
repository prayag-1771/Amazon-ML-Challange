"""France number-missing rule (v17), applied in stage-2 predict after variant_rule / legal_add_rule.

Stage 2 is trained on US/India validation queries only. In one cell it rejects French pairs ~480x more often than on
validation: the S2/S3 record has NO house number (its S1 has one), the name core is identical (a legal form may be
added or dropped, not changed) and the street name is the same up to typos. On US/India validation that cell is
100% true matches (15,374 pairs, 1 rejection - itself a true match); in France stage 2 rejects 3.3% of 32,853.
Such a France top-1 pair is accepted when p2 >= 0.01 and exactly one France S1 carries that name and street name
(neither the query's 2nd candidate nor any other S1 ties with it). Same cell with an EQUAL house number is left
alone: validation shows stage-2 rejections there are mostly right. Analysis: work/v17/fr_cells.py, work/v17/fr_rule.py.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .config import WORK_DIR

W = str(WORK_DIR) + "/"
COLS = ["entity_id", "name_core", "legal", "nums", "addr_core", "alpha_comps"]
P2_MIN, STREET_SIM = 0.01, 85
# street types and stopwords removed before comparing street names ("55 Rue Faidherbe" vs "55 Rue de Bondues" share "rue")
GENERIC = {
    "rue", "avenue", "boulevard", "allee", "place", "impasse", "chemin", "route", "quai", "cours", "passage", "square",
    "rondpoint", "lotissement", "residence", "sente", "voie", "parvis", "esplanade", "promenade", "hameau", "lieu", "dit",
    "street", "road", "lane", "drive", "court", "circle", "highway", "parkway", "trail", "terrace", "way",
    "marg", "nagar", "colony", "sector", "block", "building", "unit", "floor", "house", "near", "opposite", "post",
    "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
    "de", "la", "le", "les", "du", "des", "d", "l", "et", "a", "au", "aux", "en", "sur", "saint", "sainte", "st",
    "the", "of", "and", "no", "number", "null", "na",
}


def _load(split, pairs):
    """pairs (s1_id, s23_id) + normalised S1 (n1, l1, u1, a1, c1) and query (nq, lq, uq, aq, cq, addr_empty) columns."""
    ids1, idq = pairs.select("s1_id").unique().lazy(), pairs.select("s23_id").unique().lazy()
    s1 = (pl.scan_parquet(W + f"norm_{split}_s1.parquet").select(COLS).rename(
        {"entity_id": "s1_id", "name_core": "n1", "legal": "l1", "nums": "u1", "addr_core": "a1", "alpha_comps": "c1"})
        .join(ids1, on="s1_id", how="semi").collect())
    q = (pl.concat([pl.scan_parquet(W + f"norm_{split}_s{k}.parquet").select(COLS + ["addr_empty"]) for k in (2, 3)]).rename(
        {"entity_id": "s23_id", "name_core": "nq", "legal": "lq", "nums": "uq", "addr_core": "aq", "alpha_comps": "cq"})
        .join(idq, on="s23_id", how="semi").collect())
    return pairs.join(s1, on="s1_id").join(q, on="s23_id")


def _street_name(col, city_col="c1"):
    """Distinctive street words: address words minus the S1's city/region words, digits, street types, stopwords."""
    return (pl.col(col).fill_null("").str.split(" ").list.set_difference(pl.col(city_col).fill_null("").str.replace_all(r"\|", " ").str.split(" "))
            .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().str.contains(r"\d") & ~pl.element().is_in(list(GENERIC))))
            .list.sort().list.join(" "))


def _cell(x):
    """Flag pairs in the rule's cell: identical name core, legal form same/added/dropped, query without house number,
    S1 with one, same street name (fuzzy)."""
    x = x.filter(pl.col("addr_empty") == 0).with_columns(
        _street_name("a1").alias("sn1"), _street_name("aq").alias("snq"),
        pl.col("u1").str.split(" ").list.first().replace("", None).alias("h1"),
        pl.col("uq").str.split(" ").list.first().replace("", None).alias("hq"))
    sim = cpdist(x["sn1"].to_list(), x["snq"].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    l1, lq = pl.col("l1").fill_null(""), pl.col("lq").fill_null("")
    ok = ((pl.col("n1") == pl.col("nq")) & ((l1 == lq) | (l1 == "") | (lq == "")) & pl.col("h1").is_not_null() & pl.col("hq").is_null()
          & (pl.col("sn1") != "") & (pl.col("st_sim") >= STREET_SIM))
    return x.with_columns(pl.Series("st_sim", sim)).with_columns(ok.alias("ok"))


def number_missing_flips(top, d, split, th, countries=("France",)):
    """top: one row per query (s1_id, s23_id, p2, country) after the other rules; d: all scored pairs (s1_id, s23_id, p2).
    th: expression giving the acceptance cutoff per row. Returns the (s1_id, s23_id) pairs to accept."""
    rej = top.filter(pl.col("country").is_in(list(countries)) & (pl.col("p2") < th) & (pl.col("p2") >= P2_MIN)).select("s1_id", "s23_id", "p2")
    if rej.height == 0:
        return rej.select("s1_id", "s23_id")
    cand = _cell(_load(split, rej.select("s1_id", "s23_id"))).filter(pl.col("ok")).select("s1_id", "s23_id", "nq", "snq")
    # a genuine tie with the query's 2nd-best candidate (same name core and street name)
    sec = (d.join(cand.select("s23_id"), on="s23_id", how="semi").sort("p2", descending=True)
           .with_columns(pl.col("p2").rank("ordinal", descending=True).over("s23_id").alias("r")).filter(pl.col("r") == 2).select("s1_id", "s23_id"))
    ties = _cell(_load(split, sec)).filter(pl.col("ok")).select("s23_id")
    cand = cand.join(ties, on="s23_id", how="anti")
    # exactly one S1 of the country with this name core and street name
    s1 = (pl.scan_parquet(W + f"norm_{split}_s1.parquet").filter(pl.col("country").is_in(list(countries)))
          .select(pl.col("entity_id").alias("o_id"), pl.col("name_core").alias("nq"), "addr_core", pl.col("alpha_comps").alias("c1"))
          .join(cand.select("nq").unique().lazy(), on="nq", how="semi").collect())
    s1 = s1.with_columns(_street_name("addr_core").alias("sno"))
    x = cand.select("s23_id", "nq", "snq").join(s1.select("o_id", "nq", "sno"), on="nq")
    x = x.with_columns(pl.Series("sim", cpdist(x["sno"].to_list(), x["snq"].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32)))
    owners = x.filter(pl.col("sim") >= STREET_SIM).group_by("s23_id").len("n_owner")
    return cand.join(owners, on="s23_id").filter(pl.col("n_owner") == 1).select("s1_id", "s23_id")


def number_missing_rule(top, d, split, th, countries=("France",)):
    """Raise p2 to the cutoff for the rule's pairs (see module docstring)."""
    flips = number_missing_flips(top, d, split, th, countries)
    print(f"France number-missing rule: {flips.height:,} accepted", flush=True)
    top = top.join(flips.with_columns(pl.lit(True).alias("nmsel")), on=["s1_id", "s23_id"], how="left")
    return top.with_columns(pl.when(pl.col("nmsel")).then(pl.max_horizontal("p2", th)).otherwise(pl.col("p2")).alias("p2")).drop("nmsel")
