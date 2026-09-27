"""Loading source TSVs / ground truth and writing submission files."""
import csv
from pathlib import Path

import polars as pl

from .config import DATA_DIR, WORK_DIR

SOURCE_SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}


def read_tsv(path: Path, schema=None) -> pl.DataFrame:
    # quote_char=None: fields are raw text and may contain stray quotes.
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        schema_overrides=schema,
        infer_schema_length=0,
        missing_utf8_is_empty_string=True,
    )


def load_source(split: str, source: int) -> pl.DataFrame:
    """Load <split>_source<source>.tsv, cached as parquet."""
    cache = WORK_DIR / f"raw_{split}_s{source}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    df = read_tsv(DATA_DIR / split / f"{split}_source{source}.tsv", SOURCE_SCHEMA)
    df = df.with_columns(pl.col(c).fill_null("") for c in df.columns)
    df.write_parquet(cache)
    return df


def load_ground_truth() -> pl.DataFrame:
    """Long format: one row per (s1_id, s23_id) positive pair."""
    cache = WORK_DIR / "gt_pairs.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    gt = gt.with_columns(pl.col("matched_entity_ids").fill_null(""))
    pairs = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m")
        .filter(pl.col("m") != "")
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("s23_id"))
    )
    pairs.write_parquet(cache)
    return pairs


def write_id_lists(path: Path, s1_ids, pairs: pl.DataFrame, col_name: str) -> None:
    """Write one row per S1 id with comma-joined S2/S3 ids.

    pairs: DataFrame with columns s1_id, s23_id (deduplicated inside).
    """
    grouped = (
        pairs.select("s1_id", "s23_id").unique()
        .sort("s23_id")
        .group_by("s1_id")
        .agg(pl.col("s23_id").str.join(","))
    )
    lookup = dict(zip(grouped["s1_id"].to_list(), grouped["s23_id"].to_list()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar=None, lineterminator="\n")
        w.writerow(["source1_entity_id", col_name])
        for s1 in s1_ids:
            w.writerow([s1, lookup.get(s1, "")])
