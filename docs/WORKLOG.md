# Work log

Newest entries go at the top. Times are IST.

## 2026-09-27

### ~19:00 v17: France number-missing rule (`work/v17/`)
- **Audit:** `fr_cells.py` groups each record's top-1 pair by name, house-number, legal-form and street relation. It compares US/India validation purity with France acceptance.
- **Reading examples showed two false alarms**, both fixed in `fr_rule.py`:
  - The street test counted "rue"/"de" as shared words. So "55 Rue Faidherbe" and "55 Rue de Bondues" looked like the same street. Those rejections are correct.
  - Compagnie/Cie is mapped to the legal form "co", so "Cousettes Compagnie" → "Cousettes Primaire" looked like a legal-form change. In France it is a swap of one real word for another.
- **Bug fixed:** an empty house number came through as `""` instead of null, so the "number missing" cell never fired.
- **Equal house number: not flipped.** France rejects 0.17% vs 0.07% on validation, and validation flip precision is 0.09.
- **Number missing on the query: flipped.** Validation purity is 1.000 (15,374 pairs, 1 rejection, which was a true match). France rejects 3.3%.
  - After removing ties (another candidate or another France business with the same name and street), 806 pairs remain.
  - These are written to `v17/fr_patch.parquet`.
- **`v17/build.py`** applies patches to any base submission. It keeps each Source-2/3 record matched at most once, keeps matches a subset of candidates, writes `MANIFEST.txt` and runs the official validator.
- **Built `work/sub17_fr_on_v15/`:** v15 plus 806 pairs. Validator PASS with `--check-ids`. The candidate file is identical to v15's.
- **Expected LB +0.00004** (+0.0003 on France's score). Not uploaded; it should go out bundled with the next change.

### ~17:40 Checked the team notes (pasted by the user) against validation
- **Confirmed:**
  - About 60% of no-candidate empty-address misses have a name shared by 11+ businesses.
  - The India cutoff 0.9 → 0.6 is worth about +0.00014.
  - The expected-F rule and per-segment thresholds are dead ends.
- **Corrected:**
  - *No-candidate empty-address loss:* 4,949 pairs, +0.0021 if all fixed, not 0.005.
  - *"Found but below cutoff" US pairs:* the 0.0035 is the gain on the US score. On the overall score it is +0.0021, and 92% of these are genuine ties.
  - *Rescue:* it accepts the right owner for only 3.3% (US) and 15.5% (India). The notes' 24%/35% is probably retrieval.
  - *"Fewer addressed matches" signal:* real but weak. It lifts two-way ties from 47% to 51%; legal form gives 60%, and both together 63%.

### ~16:30 Loss analysis against v15 (`work/analysis_0927/`)
See [analysis/2026-09-27-loss-analysis.md](analysis/2026-09-27-loss-analysis.md).
- **Ceiling:** LB about 0.9967, so 0.999 is not reachable.
- **Tested and dropped:** France region imputation, tie-breaking by match count, the expected-F rule, and a resolver for empty-address ties.
- **Measured:**
  - v16 cutoff sweep: recommend 0.7.
  - India channel retrieval recall: 61.8%.

### ~15:00 Environment on this machine
- **Project copied from the old machine.** It was moved from `E:\Programming\...` to `G:\Amazon-ML-Challenge-2026`. Not copied: `.git`, `SUBMISSIONS.md`, the filled-in documentation, `dist/` and `output/`.
- **Hardware:** GTX 1650 (4 GB), 14 GB RAM, 8 threads. That is enough for rules and LightGBM on the cached scores, not for retraining the cross-encoders or running the full pipeline.
- **Set up:**
  - `.venv` created (polars 1.44.2, lightgbm 4.7.0, rapidfuzz, scikit-learn).
  - `student_resource/utils` and `student_resource/dataset/test` extracted, for the validator.
- **Checked:** the local `work/sub1..sub15` outputs match `submissions/vN/MANIFEST.txt` byte for byte.
- **Git repo set up** with `.gitignore`: code, docs and logs only (about 2 MB), no data or model binaries.
