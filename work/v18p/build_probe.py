"""v18 probes: v17 with ONLY the France stage-2 cutoff changed (US/India byte-identical to v17).

Re-runs stage 2's decision part from the cached stage-2 scores exactly as src.stage2.stage_predict does
(argmax, shift_cells + shift_cap, variant_rule, legal_add_rule, France number-missing rule, per-country cutoff), with
T["France"] replaced, then adds the unchanged rescue and v16 India-channel pairs. The candidate file is v17's.
Nothing under work/sub17_final is read for writing; outputs go to work/v18p/<name>/.

  python v18p/build_probe.py <france_cutoff> <name>      (from work/)
  sanity: cutoff 0.75 must reproduce work/sub17_final/matching_results.tsv byte for byte
"""
import hashlib
import subprocess
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parent
sys.path.insert(0, str(ROOT / "business_entity_resolution"))
import polars as pl
from src import france_rules, stage2
from src.io_utils import load_source, write_id_lists

W = str(WORK) + "/"
fr_t, name = float(sys.argv[1]), sys.argv[2]
out = WORK / "v18p" / name
out.mkdir(parents=True, exist_ok=True)
stage2.T["France"] = fr_t  # the rules read stage2.T through replace_strict
th = pl.col("country").replace_strict(stage2.T, default=0.75)

s1_ids = load_source("test", 1)["entity_id"].to_list()
ctry = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
d = pl.read_parquet(W + "test_scores_stage2_v15.parquet").join(ctry, on="s1_id")
top = d.sort("p2", descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id", "p2", "country")
top = stage2.shift_cap(stage2.shift_cells(top, "test"), pl.read_parquet(W + "shift_R.parquet"), pl.read_parquet(W + "shift_Rp.parquet"))
top = stage2.legal_add_rule(stage2.variant_rule(top, d, "test", ["France"]), d, "test")
top = france_rules.number_missing_rule(top, d, "test", th)
m = top.filter(pl.col("p2") >= th).select("s1_id", "s23_id")
res = pl.read_parquet(W + "rescue_test_accept.parquet").select("s1_id", "s23_id")
ind = pl.read_parquet(W + "v16/india16_accept.parquet").select("s1_id", "s23_id")
m = pl.concat([m, res, ind])
assert m["s23_id"].is_unique().all()
write_id_lists(out / "matching_results.tsv", s1_ids, m, "matched_entity_ids")
# candidate file: v17's, copied unchanged (all channels are identical)
(out / "candidate_pairs.tsv").write_bytes((WORK / "sub17_final" / "candidate_pairs.tsv").read_bytes())
fr = m.join(ctry, on="s1_id").group_by("country").len().sort("country").rows()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
v17 = sha(WORK / "sub17_final" / "matching_results.tsv")
h = sha(out / "matching_results.tsv")
print(f"France cutoff {fr_t}: {m.height:,} matches {fr}; sha {h[:16]}; identical to v17: {h == v17}", flush=True)
r = subprocess.run([sys.executable, str(ROOT / "student_resource/utils/validate_submission.py"), "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(ROOT / "student_resource/dataset/test")],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout.strip().splitlines()[-1])
(out / "MANIFEST.txt").write_text(f"{h}  output/matching_results.tsv\n{sha(out / 'candidate_pairs.tsv')}  output/candidate_pairs.tsv\n")
