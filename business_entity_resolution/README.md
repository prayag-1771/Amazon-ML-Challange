# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 (S1) business record, this pipeline finds all Source 2 / Source 3 (S2/S3) records that refer to the same business. Scoring is macro F0.5 per S1 entity, singletons included.

## Pipeline

```
normalize -> embed -> block (TF-IDF + e5 embeddings, per country) -> prune (cheap LightGBM, top-6 per query)
          -> cross-encoder logit (fine-tuned e5-small) + pair + context features
          -> LightGBM -> per-query argmax + threshold -> S1 match lists
```

- **Reverse-direction resolution.** In training, every S2/S3 record matches at most one S1. Each S2/S3 record is therefore treated as a query that picks its best S1 or none. Uniqueness then holds by construction.
- **Country is only a partition key.** It is never used as a feature, so countries not seen in training (France) work unchanged.
- **The pruned set is `candidate_pairs.tsv`.** It is exactly the set of candidates the final model scores.
- **Cross-fitted cross-encoder.** Two cross-encoders are fine-tuned, one on each hash half (fold A / fold B) of the training queries. Each training query gets the logit of the model that did not see it, so the LightGBM trains on all queries without leakage. Validation and test pairs get the mean of both logits (`CROSS_FIT=True`, v6).

| Module | Purpose |
|---|---|
| `src/normalize.py` | Name/address cleaning, transliteration (anyascii), legal-form canonicalisation, state map learned from training labels |
| `src/embed.py` | `intfloat/multilingual-e5-small` (MIT) embeddings of raw `name \| address` |
| `src/blocking.py` | Word TF-IDF top-5 (CPU, `sparse_dot_topn`) ∪ embedding top-10 (GPU matmul), per country |
| `src/prune.py` | Cheap LightGBM that keeps the top-6 candidates per query |
| `src/cross_encoder.py` | e5-small fine-tuned as a pair classifier (one model per fold); cached out-of-fold `ce_logit` for train, validation and test |
| `src/features.py` | rapidfuzz name/address similarities, number/state/city agreement, name rarity, within-query context |
| `src/model.py` | LightGBM training, decision rule, macro-F0.5 threshold tuning |
| `src/metric.py` | Exact reimplementation of the competition metric |
| `src/run_pipeline.py` | Stage driver |
| `src/stage2.py` | Stage-2 collective re-ranker over the ensemble scores (v9); house-number / name-diff structure and sibling-agreement features (v10); hard-pair CE feature and shift-symmetry cap (v11); cascade candidate cut, 3-seed bagging, France variant-vocabulary and legal-form-add rules, writes `candidate_pairs.tsv` (v12) |
| `src/ce_hard.py` | Hard-pair cross-encoder: e5-small fine-tuned on the ambiguous band of training pairs, scored on uncertain validation/test pairs as a stage-2 feature (v11) |
| `src/rescue.py` | Empty-address name rescue (v13): name-only char-3-gram retrieval and LightGBM for queries with no cascade candidate |
| `src/india_addr.py` | India addressed no-candidate channel (v15): per-state address-word and phonetic-name-skeleton retrieval, 3-seed LightGBM |

## Setup

Python 3.11. The GPU step uses CUDA 12.4. The embedding stage also runs on CPU, only slower.

```bash
python -m venv .venv && .venv/Scripts/activate          # or: uv venv --python 3.11
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The e5-small weights download from the Hugging Face hub on first use (about 470 MB). This is a pretrained model, not an external data source.

## Run

Data is expected at `../student_resource/dataset/{train,test}/`. You can override the locations with the `BER_DATA`, `BER_WORK` and `BER_OUTPUT` environment variables.

```bash
cd business_entity_resolution
python -m src.run_pipeline --stage all        # or one stage: normalize | embed | block | prune | ce | train | predict
python ../student_resource/utils/validate_submission.py \
    --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv \
    --test-dir ../student_resource/dataset/test --check-ids
```

Every stage caches its outputs under `work/`, so a run can resume after an interruption.

Approximate timings on an RTX 4070 laptop GPU with 48 GB RAM:

| Stage | Time |
|---|---|
| normalize | ~8 min |
| embed | ~1 h |
| block | ~1 h |
| ce (2 folds × (fine-tune ~35 min + score ~70 min)) | ~3 h 30 min |
| prune + features + train + predict | ~1–2 h |

## Validation

Ten percent of training S1 entities are held out using a deterministic hash (`config.is_valid_expr`). The model trains only on queries whose candidates are all outside the holdout. The threshold is tuned directly on macro F0.5 over the held-out S1 entities. Results are logged to `work/experiments.csv`.

## Submission versions

`matching_results.tsv` is the only file scored on the leaderboard. The final zip also needs `candidate_pairs.tsv`, this folder and the filled `Documentation_template.md`. Each leaderboard upload is tagged `sub-vN` in git. The version log is `docs/SUBMISSIONS.md` at the repository root.

| Version | Entry point | Val F0.5 |
|---|---|---|
| v1 | `python -m src.baseline test` | 0.527 |
| v2 | `python -m src.run_pipeline --stage all` (`MODEL_NAME="lgb_v1"`, `N_TRAIN_Q=2_000_000`) | 0.9771 |
| v3 | `python -m src.run_pipeline --stage all` (`MODEL_NAME="lgb_v2"`, all train queries) | 0.9786 |
| v4 | `python -m src.run_pipeline --stage all` (`MODEL_NAME="lgb_v3"`, `USE_CE=True`: adds the `ce` stage that fine-tunes and scores the cross-encoder) | 0.9885 |
| v5 | `python -m src.run_pipeline --stage all` (`MODEL_NAME="lgb_v4a"`, `CROSS_FIT=False`: adds the label-free distractor-token features from `src/token_stats.py`, cached as `work/tokstat_{split}.parquet`) | 0.9890 |
| v6 | `python -m src.run_pipeline --stage all` (`MODEL_NAME="lgb_v5cf"`, `CROSS_FIT=True`: trains the fold-B cross-encoder too, and the LightGBM on all training queries) | 0.9893 |
| v7 | `python -m src.run_pipeline --stage all` (`MODEL_NAME="lgb_v6k"`: adds context-keyed token statistics, cached as `work/tokstat2_{split}.parquet`; threshold 0.70) | 0.9893 |
| v8 | `python work/generate_sub8.py` (equal-weight ensemble of `lgb_v5cf` and `lgb_v6k`, per-country calibrated decision thresholding: US t=0.70, India t=0.80, France t=0.70) | 0.9893 |
| v9 | `python -m src.stage2 train` then `python -m src.stage2 predict` (needs `test_scores_lgb_v5cf` / `lgb_v6k` from v6 / v7 and their validation scores; stage-2 re-ranker with per-S1 support features, US/France t=0.75, India t=0.80) | 0.9895 |
| v10 | `python -m src.stage2 train` then `python -m src.stage2 predict` (same inputs as v9; adds structure and sibling-agreement features to stage 2) | 0.9898 |
| v11 | `python -m src.ce_hard train/score`, then `python -m src.stage2 train` and `python -m src.stage2 predict` (hard-pair CE feature, no house-number / name sibling counts, shift-symmetry cap) | 0.9899 |
| v12 | `python -m src.stage2 train` then `python -m src.stage2 predict` (cascade cut p ≥ 0.001, which is also `candidate_pairs.tsv`; 3-seed stage 2; `variant_rule` and `legal_add_rule` for France) | 0.9899 |
| v13 | `python -m src.rescue train` (→ `work/rescue.lgb`), then `python -m src.stage2 predict` (v12 plus rescued pairs appended to both TSVs) | 0.9901 |
| v14 | `python -m src.stage2 train` then `python -m src.stage2 predict` (adds `ndiff`, `nd3`, `nratio` to stage 2; rescue as v13) | 0.9902 |
| v15 | `python -m src.india_addr train` (→ `work/india_addr_s{1,2,3}.lgb`), then `python -m src.stage2 predict` (v14 plus the India addressed pairs appended to both TSVs) | 0.9906 |
| v17 | v15 outputs + `cd work && python v17/build.py sub15_indiaaddr <out> v17/fr_patch.parquet` (France number-missing rule, `work/v17/fr_rule.py`; France only, so val is unchanged) | 0.9906 |
