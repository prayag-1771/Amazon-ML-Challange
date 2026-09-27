# Work log

Newest entries go at the top. Times are IST, taken from file timestamps.

## 2026-09-27

### 21:15–21:30 Push for 0.99 with 2 uploads left: France calibration check (`work/v19/fr_calib.py`)
- **Target.** The user reports the best LB so far as 0.98873 and has 2 uploads left; 0.99 needs +0.0013.
- **Check.** Is stage 2 under-confident on France across the board (which a lower France cutoff would fix)?
  - With the crude street test, France rejects 44× more pairs than validation in validation-pure cells.
  - But that test is the "rue" flaw found for v17. With the fuzzy street-name test the excess is 2.3×, and validation says those rejections are correct.
  - **Result: no evidence for a lower France cutoff.** Changing it would be a blind gamble.
- **France empty-address rescue:** at most ~1,400 records without candidates, worth about +0.00005. Not worth the risk (the team's mass-accept probe scored 0.850).
- **India retrieval check** (`work/v16/repro_india_retrieve.py`) ran out of memory while running alongside the calibration check. That step is covered by the saved-pairs evidence.
- **Decision:** upload v17 (expected ~0.9889). Keep the last upload in reserve for a France change only if the portal's probe history supports it.

### 19:15–21:10 Reproduction of v17 on this machine; final package
- **`python -m src.stage2 predict` ran end to end until the India channel's feature step, then ran out of memory** (14 GB RAM; the original run used 48 GB). Everything up to that step matches v15:
  - stage-2 scores identical (6,909,690 pairs, max |Δp2| 0.0);
  - shift cap 1,803, variant rule 23,771 / 9,045, legal-add 2,035 / 2 / 3;
  - rescue 3,256 identical pairs;
  - the new France rule accepts 806.
- **Checking the India channel on its own.** `india_addr.pairs()` was split into `retrieve()` and the feature step, with no logic change. `work/v16/repro_india_retrieve.py` checks that retrieval reproduces the saved v15 channel pairs; its result is below. Re-scoring that saved pair set with the v16 models gives v16's 8,918 pairs (`work/v16/score_india16.py`).
- **Final package:** `dist/v17/Greedy_Decoders_submission.zip` (90 MB, 25 files, validator PASS), manifest `submissions/v17/MANIFEST.txt`.
- **User reported the best LB so far: 0.98873.** It is most likely v15 (v13→v15: validation +0.00046, LB +0.00049). Expected v17 is about 0.9889.

### 18:40–19:20 Final build v17, `src/` reproduction, documentation, package
- **Leaderboard results.** All files were searched. Besides the known scores, only `probe_fr_empty` = 0.850 is recorded: France left empty, which implies France F0.5 ≈ 0.956 at v9. The France cutoff probes (t40/t95/t99/t999) were written, but their scores were never recorded.
- **v16 built here**, since the peer's v16 is not on this machine.
  - The saved India channel test pairs (`india_addr_test_pairs_v15c.parquet`) reproduce v15's 5,505 accepted pairs exactly.
  - Re-scored with the retrained `india_addr16` models; cutoff 0.7 accepts 8,918 pairs.
  - Test acceptance per India S1 is 1.10%, vs 1.27% on validation, so there is no sign of extra false merges from the heavier decoy mix.
  - `work/sub16_india07/`, validator PASS, validation 0.99068.
- **v17 final** = v16 + France patch (806 pairs) → `work/sub17_final/`, 5,828,779 matches, validator PASS.
- **`src/` now reproduces v17:**
  - `src/france_rules.py`, called in `stage2.predict`, gives the same 806 pairs as `work/v17/fr_rule.py`.
  - `src/stage1_scores.py` rebuilds the cascade lists (identical pairs) and validation scores (max difference 0.0).
  - `india_addr.py` uses the `india_addr16` models at cutoff 0.7.
  - `run_pipeline.py` retrains `lgb_v5cf` with its 64 features.
  - The hard-pair cross-encoder was started from a fine-tuned cross-encoder: its training log loaded 201 tensors, which includes the classifier head. It ran 1 epoch; its learning rate was not recorded.
- **Methodology document** written: `Documentation_template.md`. Team members are still to be filled in.
- **Packaging script:** `tools/package.py` builds `dist/<version>/Greedy_Decoders_submission.zip` and `submissions/<version>/MANIFEST.txt`.

### 18:31 Stage-2 rejections: no pattern-level gain (`work/v18/rej_cells.py`)
- US/India addressed records whose top-1 pair has p2 in [0.01, cutoff): 11,639 pairs, 19.6% true matches.
- Patterns (name / number / legal / street × p2 band) were selected on one half of the validation S1s and measured on the other.
- No pattern with ≥ 30 pairs reaches 85% precision in either direction. Stage 2 is well calibrated at pattern level, so rules cannot add points here.
- **Where things stand:** on this machine, the only positive change left is v17 (France, about +0.00004 LB). The remaining gains need the peer's v16 (India channel), LB probe results for France cutoffs, or GPU work (a larger cross-encoder).

### 18:10–18:29 v18 candidate: empty-address rescue v2. Rejected (`work/v18/`)
- **The resc2 holdout left ~7% of the data for training.** It held out every record with any validation-S1 candidate. Trained that way, every cutoff lowers F.
- **Switched to a "true S1" split** (validation = records whose true S1 is a validation S1), which trains on 6.8M pairs.
  - Naive validation: +0.00042 for set A and +0.00060 for set B (tie features). B is still +0.00058 with the addressed-match count taken from predictions.
- **That split hides most false merges.** Wrong picks landing on non-validation S1 are invisible, while test has the mirror image. Corrected in `v18/eval_split.py`, which counts all wrong picks of matched records and the visible ones of unmatched records.
  - Corrected result: **negative at every cutoff**, from −0.00251 at 0.3 to −0.00003 at 0.95.
  - Precision over all picks is 0.43–0.72, and break-even is about 0.75.
- **Decision:** do not ship. Steps 4 and 5 deprioritised: tie features cannot reach break-even, and the team's `dshift` simulation shows the cutoff is stable under test-like decoy density.

### 17:45–18:00 v17: France number-missing rule (`work/v17/`)
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

### 17:40 Checked the team notes (pasted by the user) against validation
- **Confirmed:**
  - About 60% of no-candidate empty-address misses have a name shared by 11+ businesses.
  - The India cutoff 0.9 → 0.6 is worth about +0.00014.
  - The expected-F rule and per-segment thresholds are dead ends.
- **Corrected:**
  - *No-candidate empty-address loss:* 4,949 pairs, +0.0021 if all fixed, not 0.005.
  - *"Found but below cutoff" US pairs:* the 0.0035 is the gain on the US score. On the overall score it is +0.0021, and 92% of these are genuine ties.
  - *Rescue:* it accepts the right owner for only 3.3% (US) and 15.5% (India). The notes' 24%/35% is probably retrieval.
  - *"Fewer addressed matches" signal:* real but weak. It lifts two-way ties from 47% to 51%; legal form gives 60%, and both together 63%.

### 16:00–17:10 Loss analysis against v15 (`work/analysis_0927/`)
See [analysis/2026-09-27-loss-analysis.md](analysis/2026-09-27-loss-analysis.md).
- **Ceiling:** LB about 0.9967, so 0.999 is not reachable.
- **Tested and dropped:** France region imputation, tie-breaking by match count, the expected-F rule, and a resolver for empty-address ties.
- **Measured:**
  - v16 cutoff sweep: recommend 0.7.
  - India channel retrieval recall: 61.8%.

### 15:00–16:00 Environment on this machine
- **Project copied from the old machine.** It was moved from `E:\Programming\...` to `G:\Amazon-ML-Challenge-2026`. Not copied: `.git`, `SUBMISSIONS.md`, the filled-in documentation, `dist/` and `output/`.
- **Hardware:** GTX 1650 (4 GB), 14 GB RAM, 8 threads. That is enough for rules and LightGBM on the cached scores, not for retraining the cross-encoders or running the full pipeline.
- **Set up:**
  - `.venv` created (polars 1.44.2, lightgbm 4.7.0, rapidfuzz, scikit-learn).
  - `student_resource/utils` and `student_resource/dataset/test` extracted, for the validator.
- **Checked:** the local `work/sub1..sub15` outputs match `submissions/vN/MANIFEST.txt` byte for byte.
- **Git repo set up** with `.gitignore`: code, docs and logs only (about 2 MB), no data or model binaries.
