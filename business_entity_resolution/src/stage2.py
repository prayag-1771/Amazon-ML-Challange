"""Stage-2 collective re-ranker (v9; v10 adds name-diff / house-number structure and sibling-agreement features;
v11 drops the house-number / name sibling counts, adds the hard-pair cross-encoder score (src/ce_hard.py) and the
shift-symmetry cap, see shift_cap). v12 cuts the candidate set to first-stage p >= PCUT (cascade; the cut set is
also the published candidate list) and adds the label-free variant-vocabulary rule for unlabelled countries
(see variant_rule) and the legal-form-add rule (legal_add_rule); stage 2 is bagged over three seeds. v13 adds the empty-address name rescue (src/rescue.py).
v14 adds the exact house-number shift features (ndiff, nd3, nratio) to stage 2.
v15 adds the India addressed no-candidate channel (src/india_addr.py); v16 retrains it (india_addr16 models, cutoff 0.7).
v17 adds the France number-missing rule (src/france_rules.py) after the other France rules.

Input: first-stage scores of lgb_v5cf and lgb_v6k (p = their mean). Adds per-query and per-S1 aggregates of
those scores - e.g. how many other queries confidently pick this S1, from the same or the other source, and
the support of the competing S1 - and re-scores every pair with a second LightGBM.
Stage 2 trains on the validation queries, where first-stage scores are out-of-sample; its own quality is
measured with 5-fold CV grouped by the query's top-1 S1.

  python -m src.stage2 train     # CV report + fit on all validation queries -> work/stage2_s{seed}.lgb
  python -m src.stage2 predict   # test -> output/matching_results.tsv
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from .config import OUTPUT_DIR, WORK_DIR, is_valid_expr
from .io_utils import load_ground_truth, load_source, write_id_lists
from .metric import macro_f05
from . import france_rules, india_addr, rescue

W = str(WORK_DIR) + "/"
T = {"US": 0.75, "India": 0.80, "France": 0.75}  # India keeps v8's stricter threshold
PCUT = 0.001  # cascade: pairs below this first-stage score are dropped (halves the candidate set, CV -0.00002)

def base(split):
    if split == "valid":
        a = pl.read_parquet(W + "valid_scores_lgb_v5cf.parquet").rename({"p": "pa"})
        b = pl.read_parquet(W + "valid_scores_lgb_v6k.parquet").drop("label").rename({"p": "pb"})
        d = a.join(b, on=["s1_id", "s23_id"])
        vf = pl.read_parquet(W + "valid_feats.parquet", columns=["s1_id", "s23_id", "src", "q_addr_empty", "q_nonascii", "s1_name_cnt", "ce_logit", "nc_tset", "ac_tset"])
    else:
        a = pl.read_parquet(W + "test_scores_lgb_v5cf.parquet").rename({"p": "pa"})
        b = pl.read_parquet(W + "test_scores_lgb_v6k.parquet").rename({"p": "pb"})
        d = a.join(b, on=["s1_id", "s23_id"])
        ce = pl.read_parquet(W + "ce_test.parquet").join(
            pl.read_parquet(W + "ce_test_b.parquet").rename({"ce_logit": "ce_b"}), on=["q_row", "s1_row"])
        ce = ce.select("q_row", "s1_row", ((pl.col("ce_logit") + pl.col("ce_b")) / 2).alias("ce_logit"))
        vf = pl.read_parquet(W + "test_feats.parquet", columns=["q_row", "s1_row", "s1_id", "s23_id", "src", "q_addr_empty", "q_nonascii", "s1_name_cnt", "nc_tset", "ac_tset"])
        vf = vf.join(ce, on=["q_row", "s1_row"], how="left").drop("q_row", "s1_row")
    d = d.with_columns(((pl.col("pa") + pl.col("pb")) / 2).alias("p")).filter(pl.col("p") >= PCUT)
    return d.join(vf, on=["s1_id", "s23_id"])

NREL = {"eq": 0, "up25": 1, "dn25": 2, "far": 3, "miss": 4}

def struct_feats(d, split):
    """Distractors repeat an S1 name with extra tokens and a house number shifted by 1..25. Adds the relation of
    the first house numbers (nrel_c) and the counts of name_core tokens added / dropped by the query."""
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "name_core", "nums"]).rename(
        {"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename(
        {"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = d.select("s1_id", "s23_id").join(s1, on="s1_id", how="left").join(q, on="s23_id", how="left")
    t1 = pl.col("n1").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    tq = pl.col("nq").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False)
    b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
    diff = b - a
    nrel = (pl.when(a.is_null() | b.is_null()).then(pl.lit(4)).when(diff == 0).then(pl.lit(0))
            .when((diff > 0) & (diff <= 25)).then(pl.lit(1)).when((diff < 0) & (diff >= -25)).then(pl.lit(2))
            .otherwise(pl.lit(3)))
    x = x.select("s1_id", "s23_id", nrel.cast(pl.Int8).alias("nrel_c"),
                 tq.list.set_difference(t1).list.len().alias("add_n"), t1.list.set_difference(tq).list.len().alias("drop_n"),
                 # v14: exact signed shift (|d| > 30 folded to +-99), generator distractor offsets, number ratio
                 pl.when(diff.abs() <= 30).then(diff).otherwise(pl.when(diff > 0).then(99).otherwise(-99)).alias("ndiff"),
                 diff.is_in(D3).cast(pl.Int8).alias("nd3"), (b.cast(pl.Float64) / a.cast(pl.Float64)).alias("nratio"))
    return d.join(x, on=["s1_id", "s23_id"], how="left")

def sib_feats(d, split):
    """Sibling agreement: do the other candidate queries of the same S1 share this query's deviation? A true
    record's extra/missing tokens and house number are usually unique; a distractor family repeats them."""
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "name_core", "nums"]).rename(
        {"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename(
        {"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = d.select("s1_id", "s23_id").join(s1, on="s1_id", how="left").join(q, on="s23_id", how="left")
    sp = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    x = x.with_columns(sp("n1").alias("t1"), sp("nq").alias("tq"), pl.col("uq").str.split(" ").list.first().alias("fq"))
    x = x.with_columns(pl.col("tq").list.set_difference("t1").alias("add"), pl.col("t1").list.set_difference("tq").alias("drop"),
                       pl.len().over("s1_id").alias("nx")).with_row_index("i")
    # token counts among the S1's candidate queries
    tc = x.select("s1_id", pl.col("tq").alias("tok")).explode("tok").drop_nulls().group_by("s1_id", "tok").len("ct")
    ga = (x.select("i", "s1_id", pl.col("add").alias("tok")).explode("tok").drop_nulls().join(tc, on=["s1_id", "tok"], how="left")
          .group_by("i").agg((pl.col("ct").min() - 1).alias("sib_add_min"), (pl.col("ct").max() - 1).alias("sib_add_max")))
    gd = (x.select("i", "s1_id", "nx", pl.col("drop").alias("tok")).explode("tok").drop_nulls().join(tc, on=["s1_id", "tok"], how="left")
          .with_columns((pl.col("nx") - pl.col("ct").fill_null(0) - 1).alias("lack"))
          .group_by("i").agg(pl.col("lack").min().alias("sib_drop_min"), pl.col("lack").max().alias("sib_drop_max")))
    nc = x.filter(pl.col("fq").is_not_null()).group_by("s1_id", "fq").len("cn")
    ncq = x.group_by("s1_id", "nq").len("cq")
    x = x.join(nc, on=["s1_id", "fq"], how="left").join(ncq, on=["s1_id", "nq"], how="left").with_columns(
        pl.when(pl.col("fq").is_null()).then(-1).otherwise(pl.col("cn") - 1).alias("sib_num"),
        (pl.col("cq") - 1).alias("sib_name"))
    f = x.select("i", "s1_id", "s23_id", "sib_num", "sib_name").join(ga, on="i", how="left").join(gd, on="i", how="left")
    f = f.with_columns(pl.col("sib_add_min", "sib_add_max", "sib_drop_min", "sib_drop_max").fill_null(-1)).drop("i")
    return d.join(f, on=["s1_id", "s23_id"], how="left")

def ce_hard_feats(d, split):
    """Hard-pair cross-encoder score (null outside the band it scores), its rank and gap within the query."""
    vf = pl.read_parquet(W + f"{'valid' if split == 'valid' else 'test'}_feats.parquet", columns=["q_row", "s1_row", "s1_id", "s23_id"])
    h = vf.join(pl.read_parquet(W + f"ce_hard_s1_{split}.parquet"), on=["q_row", "s1_row"]).select("s1_id", "s23_id", "ce_h")
    d = d.join(h, on=["s1_id", "s23_id"], how="left")
    return d.with_columns(pl.col("ce_h").rank("ordinal", descending=True).over("s23_id").alias("ce_h_rank"),
                          (pl.col("ce_h") - pl.col("ce_h").max().over("s23_id")).alias("ce_h_gap"))

def feats(d):
    d = d.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("s23_id").alias("q_rank"),
        (pl.col("p") - pl.col("p").max().over("s23_id")).alias("q_gap_best"),
        pl.col("p").sort(descending=True).slice(1, 1).first().over("s23_id").fill_null(0).alias("q_p2"),
        pl.len().over("s23_id").alias("q_n"),
        pl.len().over("s1_id").alias("s1_n"),
    )
    top = d.filter(pl.col("q_rank") == 1)
    for th, nm in [(0.9, "hi"), (0.5, "mid")]:
        agg = top.filter(pl.col("p") >= th).group_by("s1_id", "src").agg(pl.len().alias("n"))
        a2 = agg.filter(pl.col("src") == 2).select("s1_id", pl.col("n").alias(f"s1_{nm}2"))
        a3 = agg.filter(pl.col("src") == 3).select("s1_id", pl.col("n").alias(f"s1_{nm}3"))
        d = d.join(a2, on="s1_id", how="left").join(a3, on="s1_id", how="left").with_columns(
            pl.col(f"s1_{nm}2").fill_null(0), pl.col(f"s1_{nm}3").fill_null(0))
        own = ((pl.col("q_rank") == 1) & (pl.col("p") >= th)).cast(pl.Int32)
        d = d.with_columns(
            (pl.when(pl.col("src") == 2).then(pl.col(f"s1_{nm}2")).otherwise(pl.col(f"s1_{nm}3")) - own).alias(f"s1_{nm}_same"),
            pl.when(pl.col("src") == 2).then(pl.col(f"s1_{nm}3")).otherwise(pl.col(f"s1_{nm}2")).alias(f"s1_{nm}_other"),
        ).drop(f"s1_{nm}2", f"s1_{nm}3")
    d = d.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("s1_id", "src").alias("s1_src_rank"),
        (pl.col("p") - pl.col("p").max().over("s1_id")).alias("s1_gap_best"),
    )
    # competitor: strongest alternative S1 in this query and its collective support
    d = d.with_columns(
        (pl.col("s1_hi_same") + pl.col("s1_hi_other")).alias("s1_hi_tot"))
    alt = d.select("s23_id", "q_rank", "s1_hi_tot", "s1_hi_same")
    r1 = alt.filter(pl.col("q_rank") == 1).select("s23_id", pl.col("s1_hi_tot").alias("r1_hi_tot"), pl.col("s1_hi_same").alias("r1_hi_same"))
    r2 = alt.filter(pl.col("q_rank") == 2).select("s23_id", pl.col("s1_hi_tot").alias("r2_hi_tot"), pl.col("s1_hi_same").alias("r2_hi_same"))
    d = d.join(r1, on="s23_id", how="left").join(r2, on="s23_id", how="left")
    d = d.with_columns(
        pl.when(pl.col("q_rank") == 1).then(pl.col("r2_hi_tot")).otherwise(pl.col("r1_hi_tot")).fill_null(-1).alias("alt_hi_tot"),
        pl.when(pl.col("q_rank") == 1).then(pl.col("r2_hi_same")).otherwise(pl.col("r1_hi_same")).fill_null(-1).alias("alt_hi_same"),
    ).drop("r1_hi_tot", "r1_hi_same", "r2_hi_tot", "r2_hi_same")
    return d

F = ["p", "pa", "pb", "ce_logit", "nc_tset", "ac_tset", "src", "q_addr_empty", "q_nonascii", "s1_name_cnt",
     "q_rank", "q_gap_best", "q_p2", "q_n", "s1_n", "s1_hi_same", "s1_hi_other", "s1_mid_same", "s1_mid_other",
     "s1_src_rank", "s1_gap_best", "s1_hi_tot", "alt_hi_tot", "alt_hi_same", "nrel_c", "add_n", "drop_n",
     "sib_add_min", "sib_add_max", "sib_drop_min", "sib_drop_max", "ce_h", "ce_h_rank", "ce_h_gap", "ndiff", "nd3", "nratio"]
# v10 also used sib_num / sib_name (siblings sharing the house number / name). On test, distractor families share
# the shifted house number, so they flipped from evidence against to evidence for a match (see shift_cap).
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7, num_threads=14)
R = 400
SEEDS = [7, 17, 27]  # stage 2 is bagged over seeds (lower retraining variance on the unlabelled country)

def fit_predict(Xtr, ytr, Xte):
    return np.mean([lgb.train({**P, "seed": sd}, lgb.Dataset(Xtr, ytr), R).predict(Xte, num_threads=14) for sd in SEEDS], axis=0)

D3 = [3, 4, 5, 7, 9, 11, 13, 21]  # house-number shifts used by the distractor generator (besides +-1, +-2)
CK = ["country", "dg", "leg", "nsame"]

def shift_cells(top, split):
    """Per top-1 pair: house-number shift bucket (dg), direction (up), legal-form relation (leg), same name_core."""
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "legal", "name_core", "nums"]).rename(
        {"entity_id": "s1_id", "legal": "l1", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "legal", "name_core", "nums"]) for i in (2, 3)]).rename(
        {"entity_id": "s23_id", "legal": "lq", "name_core": "nq", "nums": "uq"})
    x = top.join(s1, on="s1_id", how="left").join(q, on="s23_id", how="left")
    a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False)
    b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
    l1, lq = pl.col("l1").fill_null(""), pl.col("lq").fill_null("")
    x = x.with_columns((b - a).alias("dif"), (pl.col("n1") == pl.col("nq")).alias("nsame"),
                       pl.when(l1 == lq).then(pl.lit("same")).when(l1 == "").then(pl.lit("add"))
                       .when(lq == "").then(pl.lit("drop")).otherwise(pl.lit("chg")).alias("leg"))
    ad = pl.col("dif").abs()
    dg = (pl.when(ad == 1).then(pl.lit("d1")).when(ad == 2).then(pl.lit("d2")).when(ad.is_in(D3)).then(pl.lit("d3"))
          .when(ad.is_between(1, 25)).then(pl.lit("dx")).otherwise(None))
    return x.with_columns(dg.alias("dg"), (pl.col("dif") > 0).alias("up")).drop("l1", "lq", "n1", "nq", "u1", "uq", "dif")

def shift_ratio(vtop):
    """R = accepted true up-shifted / accepted down-shifted top-1 pairs per (country, dg) on validation (true
    matches drift both ways, distractors only up). Countries without labels (France) get the pooled ratio."""
    x = vtop.filter(pl.col("dg").is_not_null() & (pl.col("p2") >= pl.col("country").replace_strict(T, default=0.75)))
    r = (pl.col("label").filter(pl.col("up")).sum() / (~pl.col("up")).sum()).alias("R")
    return x.group_by("country", "dg").agg(r), x.group_by("dg").agg(r)

def shift_cap(top, R, Rp, slack=1.2, z=3.0, min_up=30):
    """Label-free guard against distractor families absent from validation. In each (country, dg, leg, nsame)
    cell, accepted up-shifted pairs should number about R * (accepted down-shifted pairs). If they exceed the
    generous bound slack * R * n_down + z * sqrt(R * n_down) + 5, only the bound's worth of highest-scoring
    up-shifted pairs stay accepted. Cells consistent with symmetry are left untouched."""
    th = pl.col("country").replace_strict(T, default=0.75)
    top = top.join(R, on=["country", "dg"], how="left").join(Rp.rename({"R": "Rp"}), on="dg", how="left").with_columns(
        pl.coalesce("R", "Rp").alias("R"), th.alias("th")).drop("Rp")
    x = top.filter(pl.col("dg").is_not_null() & (pl.col("p2") >= pl.col("th")))
    g = x.group_by(CK).agg(pl.col("up").sum().alias("nu"), (~pl.col("up")).sum().alias("nd"), pl.col("R").first())
    g = g.with_columns((slack * pl.col("R") * pl.col("nd") + z * (pl.col("R") * pl.col("nd")).sqrt() + 5).floor().cast(pl.Int64).alias("k"))
    g = g.filter((pl.col("nu") >= min_up) & (pl.col("nu") > pl.col("k")))
    cut = (x.filter(pl.col("up")).join(g.select(CK + ["k"]), on=CK)
           .with_columns(pl.col("p2").rank("ordinal", descending=True).over(CK).alias("rk"))
           .filter(pl.col("rk") > pl.col("k")).select("s23_id", pl.lit(True).alias("cut")))
    top = top.join(cut, on="s23_id", how="left").with_columns(
        pl.when(pl.col("cut")).then(pl.min_horizontal("p2", pl.col("th") - 1e-4)).otherwise(pl.col("p2")).alias("p2"))
    print(f"shift cap: {len(g)} cells, {len(cut):,} pairs rejected", g.select(CK + ["nu", "nd", "k"]).sort("nu", descending=True).head(10).rows(), flush=True)
    return top.drop("R", "th", "cut")


IN_MIN, IO_MIN, RW_MIN = 5.0, 1.5, 0.2  # variant token: swap-in rate / in-out ratio; real word: S1 frequency (per 1000 S1)

def _tok(c):
    return pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()

def _name_diff(t, s1, q):
    a, b = pl.col("u1").str.split(" ").list.first(), pl.col("uq").str.split(" ").list.first()
    t = t.join(s1, on="s1_id").join(q, on="s23_id").with_columns(_tok("n1").alias("t1"), _tok("nq").alias("tq"), (a == b).fill_null(False).alias("eq"))
    t = t.with_columns(pl.col("tq").list.set_difference("t1").alias("A"), pl.col("t1").list.set_difference("tq").alias("D"))
    return t.with_columns(pl.col("A").list.len().alias("na"), pl.col("D").list.len().alias("nd"),
                          pl.col("A").list.first().alias("at"), pl.col("D").list.first().alias("dt")).drop("t1", "tq", "A", "D", "n1", "nq", "u1", "uq")

def variant_rule(top, d, split, countries):
    """Label-free variant vocabulary. The generator's true-match name variants swap in / add a word from a small
    per-country vocabulary at an unchanged house number; such words are swapped in far more often than out
    (in/out ratio) among stage-1 top-1 pairs with an equal first house number. Real words in the same position
    swap in and out about equally. For `countries` (no validation labels, so stage 2 never saw their vocabulary):
    single-token add/swap to a variant word at an equal number is accepted (p2 >= 0.01), and a swap to a
    frequent real word that is not a variant (dissimilar, not a prefix) is rejected."""
    from rapidfuzz.distance import Levenshtein
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename(
        {"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename(
        {"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    t0 = _name_diff(d.sort("p", descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id"), s1, q)
    sw = t0.filter((pl.col("na") == 1) & (pl.col("nd") == 1) & pl.col("eq"))
    st = s1.select("country", _tok("n1").alias("tok")).explode("tok").drop_nulls().group_by("country", "tok").len("s1f")
    for z in (sw.group_by("country", pl.col("at").alias("tok")).len("sin"), sw.group_by("country", pl.col("dt").alias("tok")).len("sout")):
        st = st.join(z, on=["country", "tok"], how="full", coalesce=True)
    st = st.join(s1.group_by("country").len("ns"), on="country").with_columns(
        [(pl.col(c).fill_null(0) / pl.col("ns") * 1000).alias(c) for c in ("s1f", "sin", "sout")])
    st = st.select("country", pl.col("tok").alias("at"), pl.col("s1f").alias("af"),
                   ((pl.col("sin") >= IN_MIN) & (pl.col("sin") / (pl.col("sout") + 0.2) >= IO_MIN)).alias("var"))
    print("variant vocab", st.filter("var").group_by("country").agg(pl.col("at").sort()).sort("country").rows(), flush=True)
    x = _name_diff(top.select("s1_id", "s23_id").filter(pl.col("s1_id").is_in(s1.filter(pl.col("country").is_in(countries))["s1_id"].implode())), s1.drop("country"), q)
    x = x.join(s1.select("s1_id", "country"), on="s1_id").join(st, on=["country", "at"], how="left").with_columns(
        pl.col("var").fill_null(False), pl.col("af").fill_null(0),
        pl.struct("dt", "at").map_elements(lambda r: Levenshtein.normalized_similarity(r["dt"] or "", r["at"] or ""), return_dtype=pl.Float64).alias("sim"))
    one = pl.col("eq") & (pl.col("na") == 1) & (pl.col("nd") <= 1)
    x = x.select("s23_id", (one & pl.col("var")).alias("vsel"),
                 (one & (pl.col("nd") == 1) & ~pl.col("var") & (pl.col("sim") < 0.5) & (pl.col("af") >= RW_MIN)
                  & (pl.col("at").str.len_chars() >= 4) & ~pl.col("at").str.contains(r"\d") & ~pl.col("dt").str.starts_with(pl.col("at"))).alias("csel"))
    th = pl.col("country").replace_strict(T, default=0.75)
    top = top.join(x, on="s23_id", how="left").with_columns(pl.col("vsel").fill_null(False), pl.col("csel").fill_null(False), th.alias("th"))
    acc = pl.col("p2") >= pl.col("th")
    print(f"variant rule: {top.filter(pl.col('vsel') & ~acc & (pl.col('p2') >= 0.01)).height:,} accepted, "
          f"{top.filter(pl.col('csel') & acc).height:,} rejected", flush=True)
    p2 = (pl.when(pl.col("csel")).then(pl.min_horizontal("p2", pl.col("th") - 1e-4))
          .when(pl.col("vsel") & (pl.col("p2") >= 0.01)).then(pl.max_horizontal("p2", "th")).otherwise(pl.col("p2")))
    return top.with_columns(p2.alias("p2")).drop("vsel", "csel", "th")

def legal_add_rule(top, d, split):
    """Accept top-1 pairs where the query only adds a legal form: identical name_core and house numbers, street
    tokens overlapping > 1/3, first-stage p >= 0.9. Stage 2 demotes these in France (mostly via ce_h on the
    "SASU" spelling); on validation the cell has precision 0.999 and stage 2 already accepts all of it."""
    cols = ["entity_id", "name_core", "nums", "addr_core", "alpha_comps", "legal"]
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=cols).rename(
        {"entity_id": "s1_id", "name_core": "n1", "nums": "u1", "addr_core": "a1", "alpha_comps": "c1", "legal": "l1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=cols) for i in (2, 3)]).rename(
        {"entity_id": "s23_id", "name_core": "nq", "nums": "uq", "addr_core": "aq", "alpha_comps": "cq", "legal": "lq"})
    x = top.select("s1_id", "s23_id").join(d.select("s1_id", "s23_id", "p"), on=["s1_id", "s23_id"]).filter(pl.col("p") >= 0.9)
    x = x.join(s1, on="s1_id").join(q, on="s23_id").filter(
        (pl.col("n1") == pl.col("nq")) & (pl.col("u1") == pl.col("uq")) & (pl.col("l1").fill_null("") == "") & (pl.col("lq").fill_null("") != ""))
    tok = lambda a, c: pl.col(a).fill_null("").str.split(" ").list.set_difference(
        pl.col(c).fill_null("").str.replace_all(r"\|", " ").str.split(" ")).list.eval(pl.element().filter(~pl.element().str.contains(r"^\d")))
    x = x.with_columns(tok("a1", "c1").alias("t1"), tok("aq", "cq").alias("tq"))
    x = x.filter(pl.col("t1").list.set_intersection("tq").list.len() / pl.max_horizontal(pl.col("t1").list.len(), pl.col("tq").list.len(), 1) > 0.34)
    th = pl.col("country").replace_strict(T, default=0.75)
    top = top.join(x.select("s23_id", pl.lit(True).alias("lsel")), on="s23_id", how="left")
    print("legal-add rule:", top.filter(pl.col("lsel") & (pl.col("p2") < th)).group_by("country").len().sort("country").rows(), flush=True)
    return top.with_columns(pl.when(pl.col("lsel")).then(pl.max_horizontal("p2", th)).otherwise(pl.col("p2")).alias("p2")).drop("lsel")

def stage_train():
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
    gt = load_ground_truth()
    d = ce_hard_feats(sib_feats(struct_feats(feats(base("valid")), "train"), "train"), "valid")
    # fold of each query = hash of its top-1 S1 (keeps an S1's queries together)
    qf = d.filter(pl.col("q_rank") == 1).select("s23_id", (pl.col("s1_id").hash(3) % 5).alias("fold"))
    d = d.join(qf, on="s23_id")
    oof = np.zeros(len(d), np.float32)
    for k in range(5):
        tr, te = d["fold"] != k, d["fold"] == k
        oof[te.to_numpy()] = fit_predict(d.filter(tr).select(F).to_numpy(), d.filter(tr)["label"].to_numpy(), d.filter(te).select(F).to_numpy())
    d = d.with_columns(pl.Series("p2", oof))
    top0 = d.sort("p", descending=True).unique("s23_id", keep="first")
    top2 = d.sort("p2", descending=True).unique("s23_id", keep="first")
    for t in [0.6, 0.65, 0.7, 0.75, 0.8]:
        m0 = macro_f05(s1v["s1_id"], top0.filter(pl.col("p") >= t).select("s1_id", "s23_id"), gt, by=s1v)
        m2 = macro_f05(s1v["s1_id"], top2.filter(pl.col("p2") >= t).select("s1_id", "s23_id"), gt, by=s1v)
        print(f"t={t} base {m0['f05']:.5f} (US {m0['f05_US']:.5f} IN {m0['f05_India']:.5f})  stage2 {m2['f05']:.5f} (US {m2['f05_US']:.5f} IN {m2['f05_India']:.5f})", flush=True)
    for sd in SEEDS:
        m = lgb.train({**P, "seed": sd}, lgb.Dataset(d.select(F).to_numpy(), d["label"].to_numpy()), R)
        m.save_model(W + f"stage2_s{sd}.lgb")
    print(sorted(zip(F, m.feature_importance("gain").round()), key=lambda x: -x[1]))
    d.select("s1_id", "s23_id", "p", "p2", "label").write_parquet(W + "valid_scores_stage2.parquet")
    vtop = shift_cells(top2.select("s1_id", "s23_id", "p2", "label").join(s1v, on="s1_id"), "train")
    Rs, Rp = shift_ratio(vtop)
    Rs.write_parquet(W + "shift_R.parquet"); Rp.write_parquet(W + "shift_Rp.parquet")
    vc = shift_cap(vtop, Rs, Rp)  # on validation the cap should (almost) never fire
    th = pl.col("country").replace_strict(T, default=0.75)
    for nm, x in (("per-country T", vtop), ("+ shift cap", vc)):
        mm = macro_f05(s1v["s1_id"], x.filter(pl.col("p2") >= th).select("s1_id", "s23_id"), gt, by=s1v)
        print(f"{nm}: {mm['f05']:.5f} (US {mm['f05_US']:.5f} IN {mm['f05_India']:.5f})", flush=True)


def stage_predict():
    t0 = time.time()
    s1_ids = load_source("test", 1)["entity_id"].to_list()
    s1_country = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    d = ce_hard_feats(sib_feats(struct_feats(feats(base("test")), "test"), "test"), "test")
    assert d["ce_logit"].null_count() == 0
    X = d.select(F).to_numpy()
    p2 = np.mean([lgb.Booster(model_file=W + f"stage2_s{sd}.lgb").predict(X, num_threads=14) for sd in SEEDS], axis=0)
    d = d.with_columns(pl.Series("p2", p2)).join(s1_country, on="s1_id")
    d.select("s1_id", "s23_id", "p", "p2").write_parquet(W + "test_scores_stage2_v15.parquet")
    top = d.sort("p2", descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id", "p2", "country")
    top = shift_cap(shift_cells(top, "test"), pl.read_parquet(W + "shift_R.parquet"), pl.read_parquet(W + "shift_Rp.parquet"))
    top = legal_add_rule(variant_rule(top, d, "test", ["France"]), d, "test")
    top = france_rules.number_missing_rule(top, d, "test", pl.col("country").replace_strict(T, default=0.75))  # v17
    matches = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T, default=0.75)).select("s1_id", "s23_id")
    res = rescue.predict("test", d).select("s1_id", "s23_id")  # empty-address queries with no cascade candidate
    res.write_parquet(W + "rescue_test_accept.parquet")
    cand = pl.concat([d.select("s1_id", "s23_id"), res])
    ind = india_addr.predict("test", cand).select("s1_id", "s23_id")  # India addressed queries with no cascade / rescue candidate
    ind.write_parquet(W + "india_addr_test_accept.parquet")
    matches = pl.concat([matches, res, ind])
    cand = pl.concat([cand, ind])
    write_id_lists(OUTPUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    write_id_lists(OUTPUT_DIR / "candidate_pairs.tsv", s1_ids, cand, "candidate_entity_ids")
    print(f"test: {len(cand):,} candidates ({len(cand) / len(s1_ids):.2f}/S1), {len(matches):,} matches ({len(res):,} rescued, {len(ind):,} India addressed), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    {"train": stage_train, "predict": stage_predict}[sys.argv[1]]()
