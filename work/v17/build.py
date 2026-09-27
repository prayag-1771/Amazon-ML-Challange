"""Apply pair patches to an existing submission and validate it.

  python v17/build.py <base_dir> <out_dir> <patch.parquet> [<patch.parquet> ...]     (run from work/)

base_dir holds matching_results.tsv + candidate_pairs.tsv (e.g. sub15_indiaaddr). Each patch is a parquet with
s1_id, s23_id. A patch pair is added only if its S2/S3 record is not already matched (each record matches at most
one S1), and it is also added to the candidates, so matches stay a subset of candidates. Writes both TSVs,
MANIFEST.txt (SHA-256) and build.log into out_dir, then runs the official validator with --check-ids.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parent
sys.path.insert(0, str(ROOT / "business_entity_resolution"))
import polars as pl
from src.io_utils import write_id_lists


def read_pairs(path, col):
    t = pl.read_csv(path, separator="\t", quote_char=None, schema_overrides={col: pl.Utf8}).with_columns(pl.col(col).fill_null(""))
    ids = t["source1_entity_id"].to_list()
    pairs = t.with_columns(pl.col(col).str.split(",")).explode(col).filter(pl.col(col) != "").select(
        pl.col("source1_entity_id").alias("s1_id"), pl.col(col).alias("s23_id"))
    return ids, pairs


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    base, out, patches = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    log = []
    say = lambda s: (print(s, flush=True), log.append(s))
    s1_ids, m = read_pairs(base / "matching_results.tsv", "matched_entity_ids")
    _, c = read_pairs(base / "candidate_pairs.tsv", "candidate_entity_ids")
    say(f"base {base.name}: {m.height:,} matches, {c.height:,} candidates, {len(s1_ids):,} S1 rows")
    valid_s1 = pl.DataFrame({"s1_id": s1_ids})
    for p in patches:
        a = pl.read_parquet(p).select("s1_id", "s23_id").unique()
        a = a.join(valid_s1, on="s1_id", how="semi")
        new = a.join(m.select("s23_id"), on="s23_id", how="anti").unique("s23_id", keep="first")
        in_c = new.join(c, on=["s1_id", "s23_id"], how="semi").height
        say(f"patch {Path(p).name}: {a.height:,} pairs, {new.height:,} added (records not yet matched), {in_c:,} already candidates")
        m = pl.concat([m, new])
        c = pl.concat([c, new]).unique()
    assert m["s23_id"].is_unique().all(), "an S2/S3 record matched to two S1"
    assert m.join(c, on=["s1_id", "s23_id"], how="anti").height == 0, "match outside candidates"
    say(f"out {out.name}: {m.height:,} matches, {c.height:,} candidates ({c.height / len(s1_ids):.3f}/S1)")
    write_id_lists(out / "matching_results.tsv", s1_ids, m, "matched_entity_ids")
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, c, "candidate_entity_ids")
    manifest = [f"{sha(out / f)}  output/{f}" for f in ("matching_results.tsv", "candidate_pairs.tsv")]
    (out / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    say("\n".join(manifest))
    r = subprocess.run([sys.executable, str(ROOT / "student_resource/utils/validate_submission.py"), "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(ROOT / "student_resource/dataset/test"), "--check-ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    say(r.stdout.strip() + (("\n" + r.stderr.strip()) if r.stderr.strip() else ""))
    (out / "build.log").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
