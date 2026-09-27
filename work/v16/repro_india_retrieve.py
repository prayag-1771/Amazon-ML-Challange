"""Reproduction check for the India channel's test candidates (the full stage-2 predict runs out of memory on a 14 GB
machine in the channel's feature step).

src.india_addr.retrieve('test', cand) with cand = cascade pairs + rescue accepts (exactly what stage2.stage_predict
passes) must give the pair set of india_addr_test_pairs_v15c.parquet with the same asim / nsim. The channel's
features are deterministic functions of these pairs, and v16/score_india16.py re-scored that file with the v16
models (8,918 pairs at 0.7), so equality here closes the reproduction of the final version.

  python v16/repro_india_retrieve.py      (from work/)
"""
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parent / "business_entity_resolution"))
import polars as pl
from src import india_addr as ia

W = str(WORK) + "/"
cand = pl.concat([pl.read_parquet(W + "cascade_test_p001.parquet").select("s1_id", "s23_id"),
                  pl.read_parquet(W + "rescue_test_accept.parquet").select("s1_id", "s23_id")])
d, s1, q = ia.retrieve("test", cand)
d = (d.select("q_row", "s1_row", "asim", "nsim")
     .join(s1.select("s1_row", pl.col("entity_id").alias("s1_id")), on="s1_row")
     .join(q.select("q_row", pl.col("entity_id").alias("s23_id")), on="q_row").select("s1_id", "s23_id", "asim", "nsim"))
del s1, q
old = pl.read_parquet(W + "india_addr_test_pairs_v15c.parquet", columns=["s1_id", "s23_id", "asim", "nsim"])
j = d.join(old, on=["s1_id", "s23_id"], suffix="_old")
print(f"retrieved {d.height:,} pairs | saved v15 pairs {old.height:,} | in both {j.height:,} | "
      f"max |d asim| {(j['asim'] - j['asim_old']).abs().max()} | max |d nsim| {(j['nsim'] - j['nsim_old']).abs().max()}")
print("IDENTICAL" if d.height == old.height == j.height else "DIFFERENT")
