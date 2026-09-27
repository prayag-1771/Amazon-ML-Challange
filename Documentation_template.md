# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Greedy_Decoders
**Team Members:** [to be filled in by the team]
**Submission Date:** 2026-09-27 (final version v17)

---

## 1. Executive Summary

We treat every Source-2/3 record as a query that picks at most one Source-1 business, or none.

- **Candidates** come from a union of per-country TF-IDF and multilingual e5-small embedding retrieval, cut down by a cheap LightGBM.
- **Pair scoring** uses two first-stage LightGBMs that combine a cross-fitted, fine-tuned e5-small cross-encoder with name, address, house-number and label-free token-statistic features.
- **A second-stage re-ranker** uses collective evidence: how many other records confidently pick the same business, how house numbers shift, and whether a record's siblings agree.

France has no labels, so the France-specific rules are label-free statistics measured on the test data itself.

Final validation macro F0.5 on held-out US/India businesses is **0.9907**, up from 0.527 for an exact-key baseline. The best public leaderboard score recorded before the final version is 0.98824.

---

## 2. Methodology

### 2.1 Problem Analysis

**Data.**

| | Source 1 | Source 2 | Source 3 | Countries |
|---|---|---|---|---|
| Train | 2.21M | 5.03M | 5.29M | US 60%, India 40% of S1 |
| Test | 1.73M | 4.89M | 5.08M | France 15.0%, India 46.8%, US 38.3% |

The training ground truth has 7.64M matched pairs.

**Findings that shaped the design:**

- **Each Source-2/3 record matches at most one Source-1 business.** We therefore resolve from the S2/S3 side: each record picks its best business or none, so uniqueness holds by construction.
- **Decoy records.** The generator adds records that copy a business's name, add or swap one word ("holding", "participations", "groupe"), and shift the house number by 1–25 (mostly up). Genuine variant records show abbreviations, typos, transliterations, dropped or added legal forms, reordered address parts, and dropped address fields.
- **Missing values mark genuine variants, not decoys.**
  - Records with an empty address are true matches 97.7% of the time (overall rate 74%).
  - Records with no house number are true matches 97% of the time.
  - Decoys always keep a full address.
- **About 3% of S2/S3 records have an empty address.** Half of all Source-1 names are shared by at least two businesses, so an empty-address record often cannot be assigned. This is the largest single source of error (Section 5).
- **France differs from training.**
  - 32% of French S2/S3 addresses have no region; in training this happens 0–0.1% of the time. We checked that it does not hurt: acceptance is 59.6% without a region and 59.5% with one.
  - French legal forms (SARL, SAS, SASU, EURL, SCI) and French descriptor words never occur in training.
- **The test set has more decoys.** There are about 5.8 S2/S3 records per S1 business in test versus 4.7 in training, while matched pairs per business are similar (about 3.4). Precision therefore matters more on test than validation suggests.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking by embedding and TF-IDF retrieval, then a fine-tuned transformer cross-encoder, then gradient-boosted pair classification, then collective second-stage re-ranking, then label-free rules for the unlabelled country.

**Core Innovations:**
1. **Cross-fitted cross-encoder.** e5-small is fine-tuned on each half of the training records, so every pair gets an out-of-fold logit. This was the largest single gain (+0.010 F0.5).
2. **Label-free token statistics.** For each extra name word, we compute on each split, test included, how often it comes with an equal house number. This transfers to France without labels.
3. **Collective stage-2 re-ranker** with house-number-shift structure.
4. **Label-free safeguards for France.** Each is measured on the test data and checked against US/India validation:
   - *shift-symmetry cap:* true matches drift up and down in house number about equally, decoys drift only up;
   - *variant-vocabulary rule:* the French words the generator uses for true variants;
   - *legal-form-add rule:* a pair whose only difference is an added legal form;
   - *number-missing rule:* a pair whose only difference is a missing house number.

```
normalize ─► embed (e5-small) ─► block: TF-IDF top-5 ∪ e5 top-10, per country ─► prune (LightGBM, top-6)
   ─► features: cross-encoder logit (2 folds) + pair / context / token-statistic features
   ─► stage 1: LightGBM lgb_v5cf (64 features) and lgb_v6k (70 features), mean p
   ─► cascade cut p ≥ 0.001 (= candidate_pairs.tsv) ─► stage 2: 3-seed LightGBM re-ranker (37 features)
   ─► per-record argmax, per-country cutoff ─► label-free rules
   ─► + empty-address name rescue (US/India) + India addressed no-candidate channel ─► matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

**Blocking keys.** Retrieval always runs within one country, which acts only as a partition key.

| Channel | Input text | Top-K | Pair recall (train) |
|---|---|---|---|
| **Word TF-IDF** | normalized core name + core address + state; rare words only (min_df 2, max_df 2%) | 5 per record | 95.0% |
| **Embeddings** | `intfloat/multilingual-e5-small` (MIT), raw `name \| address`, so native scripts are kept | 10 per record (GPU matmul) | 98.0% |
| **Union** | – | ≈13.5 per record | **98.5%** |

**Pruning.** A cheap LightGBM on the two retrieval similarities and ranks plus three rapidfuzz scores keeps the top 6 per record with p0 ≥ 0.002. This leaves 13.2M test pairs at 98.3% pair recall.

**Cascade cut.** Stage 2 scores only pairs whose first-stage score is at least 0.001. This halves the set to 6.91M test pairs (3.99 per S1, 1.06 per record) with no loss in validation F0.5. Candidate recall on validation: India 97.6%, US 98.8%.

**Add-on retrieval for records that blocking misses:**
- **Empty-address name rescue (US, India):** char-3-gram TF-IDF on the name only, top 10, scored by a 20-feature LightGBM. The top-1 pair is accepted at score ≥ 0.9.
- **India addressed channel:** per state, the top 30 by address-word TF-IDF plus the top 30 by a phonetic name skeleton (vowels dropped, ph→f, c→k/s and similar), scored by a 33-feature, 3-seed LightGBM. The top-1 pair is accepted at score ≥ 0.7. It targets transliterated names such as "Shri Ganesh" vs "Sree Ganesha" whose address overlaps only partly.

**Candidate pairs generated:** `candidate_pairs.tsv` holds **6,921,864** pairs (3.995 per S1). It is exactly the set the final models score, plus the two channels' pairs, and every match is a candidate.

**How we ensured true matches were not lost:**
- two complementary retrieval channels, one lexical and one semantic;
- a learned prune instead of a fixed cutoff;
- channel recall measured at every stage;
- dedicated channels for the two measured gap populations (empty-address records, and India records with partial addresses).

---

## 4. Matching Model

**Cross-encoder.** `intfloat/multilingual-e5-small` (MIT, 118M parameters) is fine-tuned as a pair classifier.
- **Input:** the sentence pair `"<S1 name> | <S1 address>"` and `"<record name> | <record address>"`, at most 128 tokens.
- **Training:** 1.5M pairs per fold, one epoch, AdamW with learning rate 5e-5, 5% warm-up, bf16.
- **Cross-fitting:** two models, one per hash half of the training records. Each training pair gets the logit of the model that did not see it; validation and test pairs get the mean of both.
- **Quality:** validation AUC 0.9962, against 0.9691 for the prune score.
- **Hard-pair cross-encoder:** a second e5-small, fine-tuned from the first on training pairs in the ambiguous band (|logit| < 6), scores uncertain validation and test pairs as a stage-2 feature.

**Features used:**
- **Name:** rapidfuzz ratio, token-set, token-sort, partial, Jaro-Winkler and no-space ratio on the core name; ratio on the full name; trade-name ("doing business as") match; exact core-name match; legal-form agreement; name lengths; name rarity (how many S1 of the country share the name, and share it within the state).
- **Address:** token-set, ratio, partial and token-sort on the core address; token Jaccard; city-component Jaccard; state agreement (state map learned from training labels); house-number features (set Jaccard, first-number match, minimum relative difference, counts); address-empty flags.
- **Token statistics (label-free):** for the name words found on only one side, their rate of house-number agreement per country among each record's best pair, computed separately on every split. Keyed also by context: added vs swapped, first / middle / last position. This flags decoy words such as "holding" (agreement 0.02) against harmless variants (about 0.7).
- **Context:** rank and gaps of the prune score, the cross-encoder logit and the name similarity within the record's candidates and within the business's records.
- **Stage 2 (37 features):**
  - the ensemble score and its within-record ranks and gaps;
  - collective support: how many other records confidently pick this business, from the same and from the other source, and the same counts for the competing business;
  - house-number shift structure: equal / up / down / far, the exact signed shift, whether the shift is one the decoy generator uses, and the number ratio;
  - words added or dropped relative to the business name, and whether sibling records share those words;
  - the hard-pair cross-encoder score.

**Model type:**
- **Stage 1:** LightGBM, binary objective (127 leaves, learning rate 0.08, early stopping). Two models: `lgb_v5cf` (64 features) and `lgb_v6k` (70 features, adding the context-keyed token statistics), averaged. Trained on 10.8M pairs from all training records that touch no held-out business.
- **Stage 2:** LightGBM (31 leaves, 400 rounds), bagged over 3 seeds. Trained on validation records, where the first-stage scores are out-of-sample. Quality is measured by 5-fold CV grouped by each record's top-1 business.

**Decision rule.** Each record goes to its highest-scoring business if the stage-2 score clears the country cutoff: US 0.75, India 0.80, France 0.75. Cutoffs were chosen for macro F0.5 on validation, and checked against simulations with the test set's higher decoy share.

**Label-free rules for France**, each checked against US/India validation:

| Rule | What it does | Evidence |
|---|---|---|
| Shift-symmetry cap | Caps accepted up-shifted pairs per (country, shift bucket, legal relation, same name) cell at the level the down-shifted pairs imply | Never fires on validation; rejects about 1.8k pairs on test, mostly French |
| Variant-vocabulary rule | Accepts single-word adds/swaps to the generator's French variant words at an equal house number, and rejects swaps to frequent real words. The vocabulary comes from test swap-in/out rates: associés, développement, fils, france, groupe, services | Same method on US/India recovers their known vocabulary. LB 0.9848 → 0.9879 |
| Legal-form-add rule | Accepts pairs whose only difference is an added legal form (mostly SASU) | 0.999 precision on validation |
| Number-missing rule (v17) | Accepts pairs whose only difference is a missing house number, with an identical core name, the same street name, and exactly one French business owning that name and street. Adds 806 pairs | 100% true on validation (15,374 pairs); stage 2 rejects 3.3% of the French cell vs 0.007% on validation |

**Threshold selection method:** macro F0.5 optimisation on the held-out 10% of S1 businesses, with the exact competition metric reimplemented in `src/metric.py`.

---

## 5. Results & Error Analysis

**Validation.** 10% of training S1 businesses are held out by a deterministic hash. No held-out label is ever used in training: a record is used for training only if none of its candidates is a held-out business.

| Version | Change | Val F0.5 | Public LB |
|---|---|---|---|
| v1 | Exact-key baseline | 0.527 | – |
| v2 | Retrieval + prune + LightGBM | 0.9771 | – |
| v4 | + Cross-encoder | 0.9885 | 0.9822 |
| v5 | + Label-free token statistics | 0.9890 | 0.9836 |
| v6 | + Cross-fitting | 0.9893 | 0.9841 |
| v7 | + Context-keyed token statistics | 0.9893 | 0.9844 |
| v9 | + Stage-2 re-ranker | 0.9895 | 0.9848 |
| v11 + probe | + Hard-pair cross-encoder, shift cap; France variant rule | 0.9899 | 0.9879 |
| v13 | + Empty-address rescue | 0.9901 | **0.98824** |
| v15 | + India addressed channel | 0.99055 | – |
| **v17 (final)** | India channel retrained (cutoff 0.7) + France number-missing rule | **0.99068** (US 0.99087, India 0.99041) | – |

- **F0.5 Score (macro, validation):** 0.9907. Micro precision is 0.999 and micro recall 0.972.

**Where the remaining validation loss comes from** (v15; the gain if each category were fixed completely):

| Category | Pairs | F0.5 gain if fixed |
|---|---|---|
| Empty address, name shared by ≥ 2 businesses | 13,912 | 0.0057 |
| Addressed, missed by blocking | 3,065 | 0.0014 |
| Addressed, rejected below cutoff | 2,408 | 0.0010 |
| Empty address, unique name | 1,783 | 0.0007 |
| False merges | 776 | 0.0009 |

- **Ceiling.** Empty-address records whose name is shared by several businesses carry no information that identifies the owner. We tested four ways to break the tie: legal form, the business's number of other matches, a trained resolver, and per-business expected-F decoding. None beats the precision F0.5 requires (about 0.75, because a false merge costs about 3× a missed match). We estimate this caps the test score at about 0.9967.
- **Common false positives (wrong merges):** records that belong to a same-name business elsewhere (360 on validation); decoys with a shifted house number and an extra word matched to a business that has other matches (389); decoys matched to businesses that have no match (27).
- **Common false negatives (missed matches):**
  - empty-address records with a shared name (the dominant group);
  - Indian records with transliterated names and partial addresses that blocking misses;
  - records with several simultaneous changes (typo + word swap + missing number) that the model scores just below the cutoff.

---

## 6. Conclusion

**Approach.** Resolving from the Source-2/3 side, a cross-fitted transformer cross-encoder inside gradient boosting, and a collective re-ranker took macro F0.5 from 0.53 to 0.9907 on held-out businesses.

**France.** Because France has no labels, every France-specific decision is a label-free statistic measured on the test data and checked on US/India validation first. The largest leaderboard gain came from one such rule (+0.003).

**Lessons:**
- *Missing values tell you something:* in this data they mark genuine variants, not decoys.
- *Most remaining loss is irreducible:* ambiguous empty-address records account for most of it.
- *Validation design matters as much as the model.* A rescue channel that looked like +0.0006 on a natural-looking split was negative once false merges on non-held-out businesses were counted, so we did not ship it.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains all source in `src/`, the `README.md` with the exact command sequence, and `requirements.txt` (Python 3.11, polars, LightGBM, rapidfuzz, scikit-learn, sparse_dot_topn, sentence-transformers / transformers, torch CUDA 12.4).

**Entry points to reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`:**
1. `python -m src.run_pipeline --stage all --model lgb_v6k`, then `--stage train` and `--stage predict --model lgb_v5cf`
2. `python -m src.stage1_scores`
3. `python -m src.ce_hard train s1 ../work/ce_e5s 1 5e-5`, then `python -m src.ce_hard score s1`
4. `python -m src.stage2 train`, `python -m src.rescue train`, `python -m src.india_addr train`
5. `python -m src.stage2 predict`

| Module | Role |
|---|---|
| `normalize.py` | Cleaning, transliteration (anyascii), legal forms, state map |
| `embed.py`, `blocking.py`, `prune.py` | Retrieval and pruning |
| `cross_encoder.py`, `ce_hard.py` | Cross-encoders |
| `features.py`, `token_stats.py` | Pair and label-free token features |
| `model.py`, `metric.py`, `run_pipeline.py` | Stage-1 training, decision rule, exact metric, driver |
| `stage1_scores.py` | Stage-1 files for stage 2 |
| `stage2.py` | Re-ranker, shift cap, France variant and legal-add rules |
| `france_rules.py` | Number-missing rule |
| `rescue.py`, `india_addr.py` | Add-on channels |

The full run takes about 10–12 hours on an RTX 4070 laptop GPU with 48 GB RAM (stage timings are in the README). Every stage caches its output under `work/`.

**Fair play.** No external data, APIs or lookups are used. The only pretrained model is `intfloat/multilingual-e5-small` (MIT licence, 118M parameters), used for embeddings and as the base of both cross-encoders. All dictionaries (state map, legal forms, variant vocabulary, token statistics) are learned from the provided training labels or are label-free statistics of the provided data.

### B. Additional Results

**Stage-2 cutoff under test-like decoy density.** Validation decoys were replicated 1.4–2.0× to match the test set's share of unmatched records. The best cutoff stays at 0.75–0.80, and moving it changes F0.5 by at most 0.00003.

**India addressed channel cutoff (v16 models, validation):**

| Cutoff | F0.5 gain | Precision |
|---|---|---|
| 0.5 | +0.00015 | 0.904 |
| 0.6 | +0.00015 | 0.931 |
| **0.7 (chosen)** | +0.00013 | 0.949 |
| 0.9 | +0.00002 | 0.994 |

0.7 was chosen as the safer option because the test set has more decoys.
