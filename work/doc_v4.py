from pathlib import Path

p = Path("Documentation_template.md"); s = p.read_text(encoding="utf-8")
rep = [
("**Submission Date:** 2026-09-25 (submission v3)", "**Submission Date:** 2026-09-25 (submission v4)"),
("""A cheap LightGBM pre-ranker cuts the candidates to about 1.3 per query. A second LightGBM with 48 name, address, number and context features then scores each pair. We assign a query to its highest-scoring S1 when that score clears a threshold, chosen to maximise macro F0.5 on a held-out 10% of S1 entities.

Validation macro F0.5 is **0.979** (US 0.981, India 0.975).""",
"""A cheap LightGBM pre-ranker cuts the candidates to about 1.3 per query. Each remaining pair is then read by a **cross-encoder**: e5-small fine-tuned to classify whether the two records are the same business. A second LightGBM combines the cross-encoder's score with 48 name, address, number and context features. We assign a query to its highest-scoring S1 when that score clears a threshold, chosen to maximise macro F0.5 on a held-out 10% of S1 entities.

Validation macro F0.5 is **0.9885** (US 0.9893, India 0.9874)."""),
("**Approach Type:** Blocking + classifier (hybrid: multilingual embedding retrieval, TF-IDF retrieval, and gradient-boosted pair classification)",
 "**Approach Type:** Blocking + classifier (hybrid: multilingual embedding retrieval, TF-IDF retrieval, a fine-tuned transformer cross-encoder, and gradient-boosted pair classification)"),
("""          ─► pair + context features ─► LightGBM ─► per-query argmax, p ≥ t ─► matching_results.tsv""",
"""          ─► cross-encoder logit (fine-tuned e5-small) + pair + context features
          ─► LightGBM ─► per-query argmax, p ≥ t ─► matching_results.tsv"""),
("**Features used (48):**", """**Cross-encoder (v4).** `intfloat/multilingual-e5-small` (MIT, 118M parameters) is fine-tuned as a binary pair classifier.
- **Input:** `"<S1 name> | <S1 address>"` and `"<query name> | <query address>"` as a sentence pair, raw text, at most 128 tokens.
- **Training data:** 1.5M labelled pairs from the pruned candidates of **fold A**, a hash half of the training queries. The negatives are hard negatives from blocking.
- **Training settings:** one epoch, AdamW with learning rate 5e-5, 5% warm-up, bf16. It takes about 35 minutes on the RTX 4070.
- **Features passed to the LightGBM:** the logit `ce_logit`, its rank within the query, and its gaps to the query's best and second-best candidates.
- **Leakage control:** the LightGBM is trained only on **fold B** queries, which the cross-encoder never saw. Validation queries (those touching held-out S1) are excluded from both folds.
- **Inference:** fp16 with length-sorted batches, about 5k pairs per second. Scoring the 13.2M test pairs takes about 45 minutes.
- **Quality:** on validation pairs, the cross-encoder's AUC is 0.9962, against 0.9691 for the prune score.

**Features used (52 in v4, 48 in v2 and v3):**"""),
("""- Trained on **all** training queries (10.8M pairs, 62% positive), about 2,000 rounds. A query is used only if none of its candidates is a held-out S1. v2 used a 2M-query sample.
- The embedding model (e5-small, MIT, 118M parameters) is the only neural model. It is well within the MIT/Apache and 8B-parameter limits.""",
"""- v4 is trained on the fold-B training queries (5.4M pairs, 62% positive). A query is used only if none of its candidates is a held-out S1. v3 used all training queries, and v2 used a 2M-query sample.
- The only neural models are the e5-small embedder and the e5-small cross-encoder (MIT, 118M parameters each). Both are well within the MIT/Apache and 8B-parameter limits."""),
("The chosen threshold is t = 0.65.", "The chosen threshold is t = 0.75 in v4 (0.65 in v2 and v3)."),
("""| **v3 same, trained on all queries** | **0.9786** | 0.9809 | 0.9751 | 0.9924 | 0.9547 | 0.9770 |""",
"""| v3 same, trained on all queries | 0.9786 | 0.9809 | 0.9751 | 0.9924 | 0.9547 | 0.9770 |
| **v4 + cross-encoder feature** | **0.9885** | 0.9893 | 0.9874 | 0.9983 | 0.9683 | 0.9940 |"""),
("- **Common false positives (wrong merges):** 10.4k on validation.",
 "The error analysis below is for v3. With the cross-encoder, v4 removes about 77% of v3's false positives (micro precision rises from 0.9924 to 0.9983) and about 30% of its false negatives (micro recall rises from 0.9547 to 0.9683).\n\n- **Common false positives (wrong merges):** 10.4k on validation."),
("Planned improvements (next versions): a cross-encoder re-ranker as an extra LightGBM feature, extra retrieval and features for empty-address queries (name-rarity-aware), and France pseudo-labelling.",
 "Planned improvements (next versions): extra retrieval and features for empty-address queries (name-rarity-aware), France pseudo-labelling, and cross-encoder scores for fold A so the LightGBM can train on all queries."),
("a learned prune, and a LightGBM that uses name, address, number and within-query context features. It takes macro F0.5 from 0.53 (exact-key baseline) to 0.979 on held-out entities.",
 "a learned prune, a fine-tuned cross-encoder, and a LightGBM that uses name, address, number and within-query context features. It takes macro F0.5 from 0.53 (exact-key baseline) to 0.9885 on held-out entities."),
("""- Records with no address are the main remaining error source.""",
 """- Reading both records jointly with a small fine-tuned transformer gave the largest single gain (+0.010 F0.5). This comes mostly from precision, which the F0.5 metric weights highly.
- Records with no address are the main remaining error source."""),
("| `src/metric.py` |", "| `src/cross_encoder.py` | Cross-encoder fine-tuning on fold A and cached `ce_logit` scoring (`work/ce_{split}.parquet`) |\n| `src/metric.py` |"),
("`normalize → embed → block → prune → train → predict`", "`normalize → embed → block → prune → ce → train → predict`"),
("The full run takes about 3–4 hours on an RTX 4070 (8 GB) with 48 GB RAM.",
 "The full run takes about 5–6 hours on an RTX 4070 (8 GB) with 48 GB RAM, including about 2 hours for cross-encoder training and scoring."),
("**Threshold sweep (v3, validation):**", """**Threshold sweep (v4, validation):**

| t | F0.5 | Precision | Recall | Singleton accuracy |
|---|---|---|---|---|
| 0.50 | 0.9877 | 0.9960 | 0.9717 | 0.9875 |
| 0.60 | 0.9883 | 0.9972 | 0.9702 | 0.9903 |
| 0.70 | 0.9885 | 0.9980 | 0.9690 | 0.9931 |
| **0.75** | **0.9885** | 0.9983 | 0.9683 | 0.9940 |
| 0.80 | 0.9884 | 0.9986 | 0.9674 | 0.9946 |
| 0.90 | 0.9879 | 0.9991 | 0.9644 | 0.9972 |

**Top features by gain (v4):** `ce_logit` (about 75% of the total gain), `p0`, `ce_qrank`, `ce_gap_best`, `num_min_rel`, `num_jacc`, `nf_ratio`, `ce_gap_2nd`, `q_nums_n`, `p0_gap_best`.

**Threshold sweep (v3, validation):**"""),
("| v3 | 2026-09-25 | 0.9786 | – | Same pipeline; LightGBM `lgb_v2` trained on all training queries |",
 "| v3 | 2026-09-25 | 0.9786 | – | Same pipeline; LightGBM `lgb_v2` trained on all training queries |\n| v4 | 2026-09-25 | 0.9885 | – | Adds a fine-tuned e5-small cross-encoder logit and its context as features; LightGBM `lgb_v3` trained on fold B, t = 0.75. Same `candidate_pairs.tsv` as v2 and v3 |"),
]
for a, b in rep:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8"); print("doc ok")

p = Path("business_entity_resolution/README.md"); s = p.read_text(encoding="utf-8")
a = '| v3 | `python -m src.run_pipeline --stage all` (`MODEL_NAME="lgb_v2"`, all train queries) | 0.9786 |'
assert a in s
s = s.replace(a, a + '\n| v4 | `python -m src.run_pipeline --stage all` (`MODEL_NAME="lgb_v3"`, `USE_CE=True`: adds the `ce` stage that fine-tunes and scores the cross-encoder) | 0.9885 |')
p.write_text(s, encoding="utf-8"); print("readme ok")
