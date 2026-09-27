"""Name / address normalisation and light address parsing.

Output columns per record (all lowercase ASCII):
  name_full   cleaned full name (legal forms kept, expanded)
  name_core   name without legal forms / honorifics / stopwords
  name_alt    trade-name part after 'doing business as' / 'trading as' / 'fka' ("" if none)
  legal       canonical legal form tokens found in the name
  addr_full   cleaned address with abbreviations expanded, state removed
  addr_core   addr_full without filler tokens
  state       canonical state/region ("" if unknown)
  nums        space-joined numeric tokens of the address (leading zeros stripped)
  alpha_comps '|'-joined address components that contain no digits (city/locality candidates)
  addr_empty  1 if the address is empty/NULL
"""
import json
import re
from collections import Counter, defaultdict

import polars as pl
from anyascii import anyascii

from .config import WORK_DIR
from .io_utils import load_source, load_ground_truth

# ---------------------------------------------------------------- dictionaries
LEGAL = {
    # canonical form: variants (post punctuation-stripping, lowercase)
    "private": ["private", "pvt", "praivet", "prayvet", "privet", "piraivet", "pryvet", "praivett",
                "pra", "pri", "pte", "pite", "pvtltd"],
    "limited": ["limited", "ltd", "li", "limted", "limitd", "mited", "lmtd", "limitad", "limitet", "limited",
                "limitedd", "limitettu", "limitte"],
    "llc": ["llc", "l l c", "l.l.c", "l.l.c."],
    "llp": ["llp", "l l p"],
    "inc": ["inc", "incorporated", "incorporation"],
    "corp": ["corp", "corporation"],
    "co": ["co", "company", "cie", "compagnie"],
    "lp": ["lp"],
    "pc": ["pc", "p c", "pllc", "p l l c"],
    "sarl": ["sarl", "s a r l"],
    "sas": ["sas", "s a s", "sasu"],
    "eurl": ["eurl"],
    "sa": ["sa"],
    "sci": ["sci"],
    "ei": ["ei"],
    "public": ["public"],
}
LEGAL_MAP = {v: k for k, vs in LEGAL.items() for v in vs}
LEGAL_MULTI = sorted([v for v in LEGAL_MAP if " " in v], key=len, reverse=True)
HONORIFIC = {"sri", "shri", "shree", "smt", "dr", "mr", "mrs", "ms", "m s", "the", "messrs"}
NAME_STOP = {"and", "of", "the", "a", "an", "de", "des", "du", "la", "le", "les", "et", "d", "l"}
ALT_MARKERS = r"\b(?:doing business as|d b a|dba|trading as|t a|formerly known as|f k a|fka|also known as|a k a|aka)\b"

ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "ln": "lane", "dr": "drive", "drv": "drive", "ct": "court", "crt": "court", "cir": "circle", "pl": "place",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "hwy": "highway", "pkwy": "parkway",
    "trl": "trail", "ter": "terrace", "terr": "terrace", "sq": "square", "mt": "mount", "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "apt": "unit", "ste": "unit", "suite": "unit", "fl": "floor",
    "flr": "floor", "twp": "township", "hts": "heights", "jn": "junction", "jct": "junction",
    "expy": "expressway", "fwy": "freeway", "cv": "cove", "pt": "point", "xing": "crossing",
    "r": "rue", "imp": "impasse", "rte": "route", "ch": "chemin", "chem": "chemin", "all": "allee",
    "rpt": "rondpoint", "pass": "passage", "qu": "quai", "crs": "cours", "sen": "sente",
    "nr": "near", "opp": "opposite", "bldg": "building", "blk": "block", "sec": "sector",
    "hno": "house", "hn": "house", "mg": "marg", "ngr": "nagar", "clny": "colony", "dist": "district",
    "tq": "taluk", "tal": "taluk", "po": "post", "vill": "village", "vil": "village",
    "bengaluru": "bangalore", "banglaore": "bangalore", "poona": "pune", "gurugram": "gurgaon",
    "nasik": "nashik", "burdwan": "bardhaman", "orissa": "odisha",
}
ADDR_STOP = {
    "no", "number", "house", "door", "flat", "plot", "unit", "floor", "near", "opposite", "c", "o",
    "the", "of", "and", "de", "la", "le", "les", "du", "des", "d", "l", "at", "post", "null", "na", "n a",
}
NULL_COMPONENTS = {"", "null", "<null>", "n/a", "na", "none", "nan", "-"}

# France: departments -> region (S1 uses regions; S2/S3 mix regions and departments).
FR_REGION = {
    "nord": "hauts de france", "pas de calais": "hauts de france", "hauts de france": "hauts de france",
    "gironde": "nouvelle aquitaine", "nouvelle aquitaine": "nouvelle aquitaine",
    "loire atlantique": "pays de la loire", "pays de la loire": "pays de la loire",
    "aisne": "hauts de france", "oise": "hauts de france", "somme": "hauts de france",
    "vendee": "pays de la loire", "maine et loire": "pays de la loire", "sarthe": "pays de la loire",
    "mayenne": "pays de la loire", "landes": "nouvelle aquitaine", "dordogne": "nouvelle aquitaine",
    "lot et garonne": "nouvelle aquitaine", "pyrenees atlantiques": "nouvelle aquitaine",
    "charente": "nouvelle aquitaine", "charente maritime": "nouvelle aquitaine",
}
STATE_MAP_PATH = WORK_DIR / "state_map.json"


# ---------------------------------------------------------------- helpers
def _translit(s: pl.Series) -> pl.Series:
    """ASCII-fold only rows that contain non-ASCII characters (anyascii is per-string Python)."""
    mask = s.str.contains(r"[^\x00-\x7F]")
    if not mask.any():
        return s
    idx = mask.arg_true()
    vals = [anyascii(x) for x in s.gather(idx).to_list()]
    return s.scatter(idx, pl.Series(vals, dtype=pl.Utf8))


def _basic(expr: pl.Expr) -> pl.Expr:
    return (
        expr.str.to_lowercase()
        .str.replace_all(r"\?", "")  # mojibake replacement chars became '?'
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"\+", " and ")
    )


def _collapse(expr: pl.Expr) -> pl.Expr:
    return expr.str.replace_all(r"\s+", " ").str.strip_chars()


def _map_tokens(expr: pl.Expr, mapping: dict) -> pl.Expr:
    return expr.str.split(" ").list.eval(pl.element().replace(mapping)).list.join(" ")


def _drop_tokens(expr: pl.Expr, drop: set) -> pl.Expr:
    return (
        expr.str.split(" ")
        .list.eval(pl.element().filter(~pl.element().is_in(list(drop)) & (pl.element() != "")))
        .list.join(" ")
    )


# ---------------------------------------------------------------- names
def normalize_names(names: pl.Series) -> pl.DataFrame:
    s = _translit(names.fill_null(""))
    df = pl.DataFrame({"raw": s}).with_columns(
        _basic(pl.col("raw"))
        .str.replace_all(r"#\s*\d+", " ")  # '#77599' style record codes
        .str.replace_all(r"\bwww\.", "")
        .str.replace_all(r"\.(com|net|org|in|co\.in|fr|biz|info|us|co)\b", " ")
        .str.replace_all(r"\bm/s\b", " ")
        .str.replace_all(r"[/]", " ")
        .alias("n0")
    )
    # strip punctuation (keep letters/digits/space); dots between single letters collapse (l.l.c -> llc)
    df = df.with_columns(
        pl.col("n0")
        .str.replace_all(r"\b([a-z])\.", "$1")
        .str.replace_all(r"[^a-z0-9 ]", " ")
        .pipe(_collapse)
        .alias("n1")
    )
    # multi-token legal variants -> single token
    for v in LEGAL_MULTI:
        df = df.with_columns(pl.col("n1").str.replace_all(rf"\b{v}\b", v.replace(" ", "")))
    df = df.with_columns(_map_tokens(pl.col("n1"), {v.replace(" ", ""): k for v, k in LEGAL_MAP.items()}).alias("n2"))
    # trade name split
    df = df.with_columns(pl.col("n2").str.replace(ALT_MARKERS, " __alt__ ").alias("n3"))
    df = df.with_columns(
        pl.col("n3").str.split("__alt__").list.get(1, null_on_oob=True).fill_null("").pipe(_collapse).alias("alt_raw"),
        pl.col("n3").str.replace("__alt__", " ").pipe(_collapse).alias("name_full"),
    )
    legal_set = set(LEGAL)
    drop = legal_set | HONORIFIC | NAME_STOP
    df = df.with_columns(
        _drop_tokens(pl.col("name_full"), drop).alias("name_core"),
        _drop_tokens(pl.col("alt_raw"), drop).alias("name_alt"),
        pl.col("name_full").str.split(" ")
        .list.eval(pl.element().filter(pl.element().is_in(list(legal_set))))
        .list.unique().list.sort().list.join(" ").alias("legal"),
    )
    # core must not be empty: fall back to full name
    df = df.with_columns(
        pl.when(pl.col("name_core") == "").then(pl.col("name_full")).otherwise(pl.col("name_core")).alias("name_core")
    )
    return df.select("name_full", "name_core", "name_alt", "legal")


# ---------------------------------------------------------------- addresses
def _addr_components(addresses: pl.Series) -> pl.DataFrame:
    s = _translit(addresses.fill_null(""))
    df = pl.DataFrame({"raw": s}).with_columns(
        _basic(pl.col("raw"))
        .str.replace_all(r"\bst\.?\s*-\s*", "saint-")  # French St-Nazaire / St.-Herblain
        .str.replace_all(r"#", " ")
        .alias("a0")
    )
    # components: split on commas, clean each component
    comps = (
        pl.col("a0").str.split(",")
        .list.eval(
            pl.element()
            .str.replace_all(r"\b([a-z])\.", "$1")
            .str.replace_all(r"[^a-z0-9/ ]", " ")
            .str.replace_all(r"\b0+(\d)", "$1")  # leading zeros
            .str.replace_all(r"\s+", " ").str.strip_chars()
        )
        .list.eval(pl.element().filter(~pl.element().is_in(list(NULL_COMPONENTS))))
    )
    return df.with_columns(comps.alias("comps")).select("comps")


def build_state_map(force: bool = False) -> dict:
    """Learn S2/S3 state-token variants -> S1 canonical state from training ground truth.

    S1 stores the state as a component (usually last). For each matched pair we align the
    S2/S3 components against the S1 state; variants with strong support/purity are kept.
    """
    if STATE_MAP_PATH.exists() and not force:
        return json.loads(STATE_MAP_PATH.read_text())
    s1 = load_source("train", 1)
    s23 = pl.concat([load_source("train", 2), load_source("train", 3)])
    c1 = _addr_components(s1["business_address"]).with_columns(s1["entity_id"], s1["country"])
    c23 = _addr_components(s23["business_address"]).with_columns(s23["entity_id"])
    # S1 canonical states: frequent last components with no digits that are *mostly* last
    # (cities such as "pune" in "..., Pune, Maharashtra" are usually not the last component).
    last1 = c1.with_columns(pl.col("comps").list.last().alias("st")).filter(~pl.col("st").str.contains(r"\d"))
    freq = last1.group_by("country", "st").len("n_last").filter(pl.col("n_last") >= 1000)
    all_cnt = (
        c1.select("country", pl.col("comps").list.unique().alias("st")).explode("st")
        .group_by("country", "st").len("n_all")
    )
    freq = freq.join(all_cnt, on=["country", "st"]).filter(pl.col("n_last") / pl.col("n_all") >= 0.6)
    canon = {(c, st) for c, st in freq.select("country", "st").iter_rows()}
    s1_state = (
        last1.join(freq.select("country", "st"), on=["country", "st"], how="semi")
        .select(pl.col("entity_id").alias("s1_id"), "st", "country", pl.col("comps").alias("comps1"))
    )
    pairs = load_ground_truth().join(s1_state, on="s1_id").join(
        c23.rename({"entity_id": "s23_id"}), on="s23_id"
    )
    ex = pairs.with_columns(
        pl.concat_list(pl.col("comps").list.first(), pl.col("comps").list.last()).alias("cand")
    ).explode("cand").filter(pl.col("cand").is_not_null() & ~pl.col("cand").str.contains(r"\d"))
    # A state variant ("mh", "mharastr") is absent from the S1 address; a city ("indore") is
    # usually present there too, so it must stay part of the address rather than become a state.
    ex = ex.with_columns(pl.col("comps1").list.contains(pl.col("cand")).alias("in_s1"))
    cnt = ex.group_by("country", "cand", "st").agg(pl.len().alias("len"), pl.col("in_s1").mean().alias("in_s1"))
    tot = cnt.group_by("country", "cand").agg(pl.col("len").sum().alias("tot"))
    best = (
        cnt.sort("len", descending=True).group_by("country", "cand").first()
        .join(tot, on=["country", "cand"])
        .filter(
            (pl.col("len") >= 200)
            & (pl.col("len") / pl.col("tot") >= 0.95)
            & ((pl.col("in_s1") <= 0.2) | (pl.col("cand") == pl.col("st")))
        )
    )
    out = defaultdict(dict)
    for country, cand, st in best.select("country", "cand", "st").iter_rows():
        out[country][cand] = st
    for country, st in canon:
        out[country].setdefault(st, st)
    # France (no training labels): departments / regions
    fr = {k: v for k, v in FR_REGION.items()}
    fr.update({k.replace(" ", "-"): v for k, v in FR_REGION.items()})
    out["France"] = fr
    STATE_MAP_PATH.write_text(json.dumps(out, indent=0, sort_keys=True))
    return out


def normalize_addresses(addresses: pl.Series, countries: pl.Series, state_map: dict) -> pl.DataFrame:
    df = _addr_components(addresses).with_columns(countries.alias("country"))
    # map every component through the (country, component) -> state dictionary
    flat = [(c, k, v) for c, m in state_map.items() for k, v in m.items()]
    smap = pl.DataFrame(flat, schema=["country", "comp", "state"], orient="row")
    df = df.with_row_index("rid")
    ex = df.select("rid", "country", "comps").explode("comps").rename({"comps": "comp"}).with_row_index("pos")
    ex = ex.join(smap, on=["country", "comp"], how="left")
    # prefer a component that *is* the canonical state ("washington, dc" -> dc, not the city
    # alias washington -> wa); among equals take the last one (usual position of the state)
    st = (
        ex.filter(pl.col("state").is_not_null())
        .with_columns((pl.col("comp") == pl.col("state")).alias("is_canon"))
        .sort("rid", "is_canon", "pos")
        .group_by("rid").agg(pl.col("state").last())
    )
    rest = (
        ex.filter(pl.col("state").is_null() & pl.col("comp").is_not_null())
        .with_columns(
            pl.col("comp").str.replace_all(r"/", " ").str.split(" ")
            .list.eval(pl.element().replace(ADDR_ABBR)).list.join(" ").alias("comp")
        )
        .group_by("rid", maintain_order=True)
        .agg(pl.col("comp"))
    )
    df = df.join(st, on="rid", how="left").join(rest, on="rid", how="left").sort("rid")
    df = df.with_columns(pl.col("comp").fill_null([]), pl.col("state").fill_null(""))
    df = df.with_columns(
        pl.col("comp").list.join(" ").pipe(_collapse).alias("addr_full"),
        pl.col("comp").list.eval(pl.element().filter(~pl.element().str.contains(r"\d"))).list.join("|").alias("alpha_comps"),
    )
    df = df.with_columns(
        _drop_tokens(pl.col("addr_full"), ADDR_STOP).alias("addr_core"),
        pl.col("addr_full").str.extract_all(r"\d+").list.join(" ").alias("nums"),
        ((pl.col("addr_full") == "") & (pl.col("state") == "")).cast(pl.Int8).alias("addr_empty"),
    )
    return df.select("addr_full", "addr_core", "state", "nums", "alpha_comps", "addr_empty")


def normalize_source(split: str, source: int, force: bool = False) -> pl.DataFrame:
    cache = WORK_DIR / f"norm_{split}_s{source}.parquet"
    if cache.exists() and not force:
        return pl.read_parquet(cache)
    raw = load_source(split, source)
    state_map = build_state_map()
    names = normalize_names(raw["business_name"])
    addrs = normalize_addresses(raw["business_address"], raw["country"], state_map)
    out = pl.concat([raw, names, addrs], how="horizontal")
    out.write_parquet(cache)
    return out


if __name__ == "__main__":
    import sys
    import time

    for split in sys.argv[1:] or ["train", "test"]:
        for src in (1, 2, 3):
            t = time.time()
            d = normalize_source(split, src, force=True)
            print(split, src, d.shape, f"{time.time() - t:.0f}s", flush=True)
