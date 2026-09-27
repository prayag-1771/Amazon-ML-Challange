"""Baseline #1: exact key match (country, name_core, state) resolved per S2/S3 query.

A query is assigned to an S1 only when exactly one S1 in the country carries its key
(ambiguous keys are skipped for precision). Candidates = all S1 sharing the key.
"""
import sys

import polars as pl

from .config import OUTPUT_DIR, is_valid_expr
from .io_utils import load_ground_truth, write_id_lists
from .metric import macro_f05
from .normalize import normalize_source

KEY = ["country", "name_core", "state"]


def run(split: str):
    s1 = normalize_source(split, 1).select(pl.col("entity_id").alias("s1_id"), *KEY)
    q = pl.concat([normalize_source(split, 2), normalize_source(split, 3)]).select(
        pl.col("entity_id").alias("s23_id"), *KEY
    )
    cand = q.join(s1, on=KEY)
    n_per_q = cand.group_by("s23_id").len("n")
    pred = cand.join(n_per_q.filter(pl.col("n") == 1), on="s23_id", how="semi").select("s1_id", "s23_id")
    return s1, cand.select("s1_id", "s23_id"), pred


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    s1, cand, pred = run(split)
    if split == "train":
        gt = load_ground_truth()
        val = s1.filter(is_valid_expr())
        print("val S1:", len(val))
        print(macro_f05(val["s1_id"], pred, gt, by=val.select("s1_id", "country")))
        print("candidate recall:", macro_f05(val["s1_id"], cand, gt)["recall_micro"])
    else:
        write_id_lists(OUTPUT_DIR / "candidate_pairs.tsv", s1["s1_id"].to_list(), cand, "candidate_entity_ids")
        write_id_lists(OUTPUT_DIR / "matching_results.tsv", s1["s1_id"].to_list(), pred, "matched_entity_ids")
        print("written", len(s1), "rows; pairs:", len(cand), len(pred))
