"""v17 France number-missing rule (refined after reading examples of the fr_cells.py flip pool).

Accept a France top-1 pair that v15 left unmatched when ALL hold:
  * name_core identical (only legal form may differ: same / added / dropped - never changed)
  * the query has NO house number while its S1 has one
  * exactly one France S1 carries this name_core and street name (unique_owner)
  * street NAME equal up to typos: address words minus S1 city/region words, digits, street types and
    stopwords (rue, avenue, de, la, ...), fuzz.ratio >= 85 on the remaining words
  * stage-2 p2 >= 0.01 (the team leaves lower-scored pairs rejected on purpose)
  * no competing candidate for the query with the same name_core and the same street name
Why: fr_cells.py showed the unrefined cells are >= 99.9% pure on US/India validation but France rejects
30-350x more of them. Examples showed two false alarms, both excluded here: same name + number on a
DIFFERENT street ("55 Rue Faidherbe" vs "55 Rue de Bondues": the word "rue" made streets look equal) and
Compagnie/Cie swaps (the normaliser maps them to the legal form "co").
With the street fixed, the equal-number cell is no longer anomalous (France rejects 0.17% vs 0.07% on validation)
and validation says those rejections are mostly right, so it is NOT flipped. The number-missing cell is: validation
purity 1.0 on 15,374 pairs with 1 rejection (a true match); France rejects 3.3% of 32,853 (~480x).

  python v17/fr_rule.py      (from work/)  -> v17/fr_patch.parquet (s1_id, s23_id, p2, cell), report on stdout
"""
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parent / "business_entity_resolution"))
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from src.config import is_valid_expr
from src.metric import macro_f05

W = str(WORK) + "/"
OUT = WORK / "v17"
COLS = ["entity_id", "name_core", "legal", "nums", "addr_core", "alpha_comps"]
GENERIC = {
    "rue", "avenue", "boulevard", "allee", "place", "impasse", "chemin", "route", "quai", "cours", "passage", "square",
    "rondpoint", "lotissement", "residence", "sente", "voie", "parvis", "esplanade", "promenade", "hameau", "lieu", "dit",
    "street", "road", "lane", "drive", "court", "circle", "boulevard", "highway", "parkway", "trail", "terrace", "way",
    "marg", "nagar", "colony", "sector", "block", "building", "unit", "floor", "house", "near", "opposite", "post",
    "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
    "de", "la", "le", "les", "du", "des", "d", "l", "et", "a", "au", "aux", "en", "sur", "saint", "sainte", "st",
    "the", "of", "and", "no", "number", "null", "na",
}
T = {"US": 0.75, "India": 0.80, "France": 0.75}


def read_pairs(path, col):
    t = pl.read_csv(path, separator="\t", quote_char=None, schema_overrides={col: pl.Utf8}).with_columns(pl.col(col).fill_null(""))
    return t.with_columns(pl.col(col).str.split(",")).explode(col).filter(pl.col(col) != "").select(
        pl.col("source1_entity_id").alias("s1_id"), pl.col(col).alias("s23_id"))


def load(split, pairs):
    ids1, idq = pairs.select("s1_id").unique().lazy(), pairs.select("s23_id").unique().lazy()
    s1 = (pl.scan_parquet(W + f"norm_{split}_s1.parquet").select(COLS).rename(
        {"entity_id": "s1_id", "name_core": "n1", "legal": "l1", "nums": "u1", "addr_core": "a1", "alpha_comps": "c1"})
        .join(ids1, on="s1_id", how="semi").collect())
    q = (pl.concat([pl.scan_parquet(W + f"norm_{split}_s{k}.parquet").select(COLS + ["addr_empty"]) for k in (2, 3)]).rename(
        {"entity_id": "s23_id", "name_core": "nq", "legal": "lq", "nums": "uq", "addr_core": "aq", "alpha_comps": "cq"})
        .join(idq, on="s23_id", how="semi").collect())
    return pairs.join(s1, on="s1_id").join(q, on="s23_id")


def street_name(col):
    """Distinctive street words: address words minus the S1's city/region words, digits, street types, stopwords."""
    return (pl.col(col).fill_null("").str.split(" ").list.set_difference(pl.col("c1").fill_null("").str.replace_all(r"\|", " ").str.split(" "))
            .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().str.contains(r"\d") & ~pl.element().is_in(list(GENERIC))))
            .list.sort().list.join(" "))


def tag(x):
    """x: pairs with S1/query columns -> rule flag `ok` (all conditions except p2 / competitor)."""
    x = x.filter(pl.col("addr_empty") == 0).with_columns(street_name("a1").alias("sn1"), street_name("aq").alias("snq"),
        # "" (no number) must become null, otherwise the number-missing case never fires
        pl.col("u1").str.split(" ").list.first().replace("", None).alias("h1"), pl.col("uq").str.split(" ").list.first().replace("", None).alias("hq"))
    sim = cpdist(x["sn1"].to_list(), x["snq"].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    l1, lq = pl.col("l1").fill_null(""), pl.col("lq").fill_null("")
    legal_ok = (l1 == lq) | (l1 == "") | (lq == "")          # same, added or dropped - not changed
    num_ok = (pl.col("h1").is_not_null() & (pl.col("h1") == pl.col("hq"))) | (pl.col("h1").is_not_null() & pl.col("hq").is_null())
    return x.with_columns(pl.Series("st_sim", sim)).with_columns(
        ((pl.col("n1") == pl.col("nq")) & legal_ok & num_ok & (pl.col("sn1") != "") & (pl.col("st_sim") >= 85)).alias("ok"),
        pl.when(pl.col("hq").is_null()).then(pl.lit("num_missing")).otherwise(pl.lit("num_eq")).alias("num_rel"),
        pl.when(l1 == lq).then(pl.lit("same")).when(l1 == "").then(pl.lit("add")).otherwise(pl.lit("drop")).alias("legal_rel"))


def competitor_free(top_ok, scores, split):
    """Drop queries whose 2nd-best candidate also has the same name_core and street name (a genuine tie)."""
    sec = (scores.join(top_ok.select("s23_id"), on="s23_id", how="semi").sort("p2", descending=True)
           .with_columns(pl.col("p2").rank("ordinal", descending=True).over("s23_id").alias("r")).filter(pl.col("r") == 2).select("s1_id", "s23_id"))
    sec = tag(load(split, sec)).filter(pl.col("ok")).select("s23_id")
    return top_ok.join(sec, on="s23_id", how="anti"), sec.height


def unique_owner(cand):
    """Keep a flip only if exactly one France S1 has the query's name_core AND its street name (the query has no
    house number, so two same-name S1 on the same street could not be told apart). Checked over all S1, not only
    the cascade candidates."""
    s1 = (pl.scan_parquet(W + "norm_test_s1.parquet").filter(pl.col("country") == "France")
          .select(pl.col("entity_id").alias("o_id"), pl.col("name_core").alias("nq"), "addr_core", pl.col("alpha_comps").alias("c1"))
          .join(cand.select("nq").unique().lazy(), on="nq", how="semi").collect())
    s1 = s1.with_columns(street_name("addr_core").alias("sno"))
    x = cand.select("s23_id", "nq", "snq").join(s1.select("o_id", "nq", "sno"), on="nq")
    x = x.with_columns(pl.Series("sim", cpdist(x["sno"].to_list(), x["snq"].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32)))
    owners = x.filter(pl.col("sim") >= 85).group_by("s23_id").len("n_owner")
    keep = cand.join(owners, on="s23_id", how="left").filter(pl.col("n_owner") == 1)
    return keep.drop("n_owner"), cand.height - keep.height


def main():
    # ---------------- validation (US/India): purity of the rule and effect of flipping
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}).filter(is_valid_expr("s1_id"))
    gt = pl.read_parquet(W + "gt_pairs.parquet").join(s1v, on="s1_id", how="semi")
    pred = pl.read_parquet(W + "pred15_v.parquet").select("s1_id", "s23_id")
    v = pl.read_parquet(W + "valid_scores_stage2.parquet", columns=["s1_id", "s23_id", "p2", "label"]).join(s1v, on="s1_id")
    vt = tag(load("train", v.sort("p2", descending=True).unique("s23_id", keep="first")))
    cell = vt.filter(pl.col("ok"))
    print(f"VALID rule cell: {cell.height:,} top-1 pairs, purity {cell['label'].mean():.5f}; by (num, legal):")
    print(cell.group_by("num_rel", "legal_rel").agg(pl.len(), pl.col("label").mean().round(5).alias("purity")).sort("len", descending=True))
    flip = cell.filter(pl.col("p2") >= 0.01).join(pred, on="s23_id", how="anti")
    base = macro_f05(s1v["s1_id"], pred, gt, by=s1v)
    m = macro_f05(s1v["s1_id"], pl.concat([pred, flip.select("s1_id", "s23_id")]), gt, by=s1v)
    print(f"VALID flips {flip.height} (precision {flip['label'].mean() if flip.height else float('nan')}), F {base['f05']:.5f} -> {m['f05']:.5f}")
    print("VALID flip precision by (num, p2 band):", flip.group_by("num_rel", pl.col("p2").cut([0.1, 0.3, 0.5])).agg(pl.len(), pl.col("label").mean().round(3)).sort("num_rel", "p2").rows())
    rej = cell.join(pred, on="s23_id", how="anti")
    print(f"VALID reject rate inside cell: {rej.height / cell.height:.5f} ({rej.height} of {cell.height}); by num_rel:",
          cell.join(pred.select("s23_id").with_columns(pl.lit(1).alias("a")), on="s23_id", how="left").group_by("num_rel").agg(pl.len(), (1 - pl.col("a").fill_null(0).mean()).round(5).alias("rej")).rows())
    del v, vt, cell
    # ---------------- test France
    frs1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}).filter(pl.col("country") == "France")
    sc = pl.read_parquet(W + "test_scores_stage2_v15.parquet", columns=["s1_id", "s23_id", "p2"]).join(frs1.select("s1_id"), on="s1_id", how="semi")
    top = sc.sort("p2", descending=True).unique("s23_id", keep="first")
    m15 = read_pairs(W + "sub15_indiaaddr/matching_results.tsv", "matched_entity_ids")
    ft = tag(load("test", top))
    cell = ft.filter(pl.col("ok"))
    acc = cell.join(m15, on=["s1_id", "s23_id"], how="semi").height
    print(f"FRANCE rule cell: {cell.height:,} top-1 pairs, v15 accepts {acc / cell.height:.4f}")
    print("FRANCE reject rate by num_rel:", cell.join(m15.select("s23_id").with_columns(pl.lit(1).alias("a")), on="s23_id", how="left").group_by("num_rel").agg(pl.len(), (1 - pl.col("a").fill_null(0).mean()).round(5).alias("rej")).rows())
    cand = cell.filter(pl.col("p2") >= 0.01).join(m15.select("s23_id"), on="s23_id", how="anti")
    print(f"FRANCE unmatched in cell with p2 >= 0.01: {cand.height:,}")
    cand, ties = competitor_free(cand, sc, "test")
    print(f"FRANCE removed as ties with the 2nd candidate: {ties}; left: {cand.height:,}")
    # Final decision (see docstring): only the number-missing case. With an equal house number, validation shows
    # stage-2 rejections inside the cell are mostly right (flip precision 0.0-0.2 below p2 0.3), so those stay rejected.
    cand = cand.filter(pl.col("num_rel") == "num_missing")
    cand, amb = unique_owner(cand)
    print(f"FRANCE number-missing flips: removed {amb} whose name + street matches more than one France S1; final flips: {cand.height:,}")
    print(cand.group_by("num_rel", "legal_rel").len().sort("len", descending=True))
    print("p2 bands:", cand.group_by(pl.col("p2").cut([0.05, 0.1, 0.3, 0.5, 0.75])).len().sort("p2").rows())
    cand.select("s1_id", "s23_id", "p2", pl.concat_str("num_rel", "legal_rel", separator="/").alias("cell"), "st_sim").write_parquet(OUT / "fr_patch.parquet")
    cand.select("s1_id", "s23_id", "p2", "n1", "nq", "sn1", "snq", "st_sim", "num_rel", "legal_rel").sample(min(40, cand.height), seed=1).write_csv(OUT / "fr_patch_sample.csv")
    print("wrote v17/fr_patch.parquet and v17/fr_patch_sample.csv")


if __name__ == "__main__":
    main()
