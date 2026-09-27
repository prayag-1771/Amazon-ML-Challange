"""Final package for the fr060 variant (v18 = v17 with the France cutoff 0.60), in a SEPARATE location.

Nothing in the repository is modified. The zip gets modified COPIES of:
  src/stage2.py              T["France"] 0.75 -> 0.60 (so the code reproduces the fr060 outputs)
  README.md                  a v18 row in the version table
  Documentation_template.md  version, France cutoff and a v18 results row
Output: dist/v18_fr060/Greedy_Decoders_submission.zip + dist/v18_fr060/MANIFEST.txt (after the official validator).

  python work/v18p/package_fr060.py      (from the repository root)
"""
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "business_entity_resolution"
OUTS = ROOT / "work" / "v18p" / "fr060"
DIST = ROOT / "dist" / "v18_fr060"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def edit(text, pairs, name):
    for a, b in pairs:
        assert text.count(a) == 1, (name, a)
        text = text.replace(a, b)
    return text


def main():
    m, c = OUTS / "matching_results.tsv", OUTS / "candidate_pairs.tsv"
    r = subprocess.run([sys.executable, str(ROOT / "student_resource/utils/validate_submission.py"), "--matching", str(m), "--candidate", str(c),
                        "--test-dir", str(ROOT / "student_resource/dataset/test"), "--check-ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout.strip())
    if r.returncode != 0:
        sys.exit("validator failed - not packaging")
    files = {}  # archive path -> bytes
    for p in sorted((CODE / "src").glob("*.py")):
        files[f"code/business_entity_resolution/src/{p.name}"] = p.read_bytes()
    s2 = (CODE / "src" / "stage2.py").read_text(encoding="utf-8")
    s2 = edit(s2, [
        ('T = {"US": 0.75, "India": 0.80, "France": 0.75}  # India keeps v8\'s stricter threshold',
         'T = {"US": 0.75, "India": 0.80, "France": 0.60}  # India keeps v8\'s stricter threshold; v18: France 0.75 -> 0.60'),
        ("v17 adds the France number-missing rule (src/france_rules.py) after the other France rules.\n",
         "v17 adds the France number-missing rule (src/france_rules.py) after the other France rules.\n"
         "v18 lowers the France cutoff to 0.60: France has 3-5x more rejected-but-uncertain top-1 records per S1 than US/India.\n"),
    ], "stage2.py")
    files["code/business_entity_resolution/src/stage2.py"] = s2.encode("utf-8")
    rd = (CODE / "README.md").read_text(encoding="utf-8")
    rd = rd.rstrip("\n") + "\n| v18 | as v17 with the France cutoff 0.60 (`T` in `src/stage2.py`); France only, so validation is unchanged | 0.9907 |\n"
    files["code/business_entity_resolution/README.md"] = rd.encode("utf-8")
    files["code/business_entity_resolution/requirements.txt"] = (CODE / "requirements.txt").read_bytes()
    doc = (ROOT / "Documentation_template.md").read_text(encoding="utf-8")
    doc = edit(doc, [
        ("**Submission Date:** 2026-09-27 (final version v17)", "**Submission Date:** 2026-09-27 (final version v18)"),
        ("country cutoff: US 0.75, India 0.80, France 0.75.",
         "country cutoff: US 0.75, India 0.80, France 0.60. France was lowered from 0.75 in v18 because, per S1, France has 3–5× more "
         "rejected-but-uncertain top-1 records (p2 0.3–0.75) than US/India and accepts fewer pairs per S1 (3.31 vs 3.37–3.39)."),
        ("| **v17 (final)** | India channel retrained (cutoff 0.7) + France number-missing rule | **0.99068** (US 0.99087, India 0.99041) | – |",
         "| v17 | India channel retrained (cutoff 0.7) + France number-missing rule | 0.99068 (US 0.99087, India 0.99041) | 0.9888 |\n"
         "| **v18 (final)** | v17 with France cutoff 0.60 (France only) | **0.99068** | see portal |"),
    ], "Documentation_template.md")
    files["Documentation_template.md"] = doc.encode("utf-8")
    files["output/matching_results.tsv"] = m.read_bytes()
    files["output/candidate_pairs.tsv"] = c.read_bytes()
    DIST.mkdir(parents=True, exist_ok=True)
    z = DIST / "Greedy_Decoders_submission.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as f:
        for arc in sorted(files):
            f.writestr(arc, files[arc])
    lines = [f"{sha(files['output/matching_results.tsv'])}  output/matching_results.tsv",
             f"{sha(files['output/candidate_pairs.tsv'])}  output/candidate_pairs.tsv", f"{sha(z.read_bytes())}  {z.name}"]
    (DIST / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {z} ({z.stat().st_size / 1e6:.0f} MB, {len(files)} files)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
