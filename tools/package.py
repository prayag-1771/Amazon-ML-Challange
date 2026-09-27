"""Build the final submission package and its manifest.

  python tools/package.py <version> <outputs_dir>        e.g.  python tools/package.py v17 work/sub17_final

Zip layout (as required by the challenge README):
  output/matching_results.tsv, output/candidate_pairs.tsv      from <outputs_dir>
  code/business_entity_resolution/{src/*.py, README.md, requirements.txt}
  Documentation_template.md
Writes dist/<version>/Greedy_Decoders_submission.zip and submissions/<version>/MANIFEST.txt (SHA-256 of both TSVs and the
zip), after running the official validator on the TSVs.
"""
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM = "Greedy_Decoders"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    version, outputs = sys.argv[1], (ROOT / sys.argv[2]).resolve()
    m, c = outputs / "matching_results.tsv", outputs / "candidate_pairs.tsv"
    r = subprocess.run([sys.executable, str(ROOT / "student_resource/utils/validate_submission.py"), "--matching", str(m), "--candidate", str(c),
                        "--test-dir", str(ROOT / "student_resource/dataset/test"), "--check-ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout.strip())
    if r.returncode != 0:
        sys.exit("validator failed - not packaging")
    code = ROOT / "business_entity_resolution"
    files = [(m, "output/matching_results.tsv"), (c, "output/candidate_pairs.tsv"),
             (code / "README.md", "code/business_entity_resolution/README.md"),
             (code / "requirements.txt", "code/business_entity_resolution/requirements.txt"),
             (ROOT / "Documentation_template.md", "Documentation_template.md")]
    files += [(p, f"code/business_entity_resolution/src/{p.name}") for p in sorted((code / "src").glob("*.py"))]
    dist = ROOT / "dist" / version
    dist.mkdir(parents=True, exist_ok=True)
    z = dist / f"{TEAM}_submission.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as f:
        for src, arc in files:
            f.write(src, arc)
    lines = [f"{sha(m)}  output/matching_results.tsv", f"{sha(c)}  output/candidate_pairs.tsv", f"{sha(z)}  {z.name}"]
    man = ROOT / "submissions" / version / "MANIFEST.txt"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text("\n".join(lines) + "\n")
    print(f"wrote {z} ({z.stat().st_size / 1e6:.0f} MB, {len(files)} files) and {man.relative_to(ROOT)}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
