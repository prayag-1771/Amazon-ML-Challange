from pathlib import Path


def patch(path, rep):
    p = Path(path); s = p.read_text(encoding="utf-8")
    for a, b in rep:
        assert s.count(a) == 1, (path, a[:80])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")


patch("Documentation_template.md", [
("**Submission Date:** 2026-09-25 (submission v4)", "**Submission Date:** 2026-09-25 (submission v5)"),
("A second LightGBM combines the cross-encoder's score with 48 name, address, number and context features.",
 "A second LightGBM combines the cross-encoder's score with 48 name, address, number and context features, plus 12 label-free \"distractor-token\" statistics computed on each split's own data."),
("Validation macro F0.5 is **0.9885** (US 0.9893, India 0.9874).", "Validation macro F0.5 is **0.9890** (US 0.9897, India 0.9878)."),
("**Features used (52 in v4, 48 in v2 and v3):**", """**Distractor-token statistics (v5, `src/token_stats.py`).** The v4 public score (0.9822) was below its validation score (0.9885). A Monte-Carlo estimate of the expected F0.5 per country, which treats the model's calibrated p as truth probabilities, pointed to France: 0.978, against about 0.992–0.994 for US and India. In France's uncertain band (p between 0.1 and 0.9), 37% of queries differ from their best S1 by a single name token. Most of these tokens are legal forms (sas vs sa, sarl vs sas) or descriptor words (holding, participations, développement, groupe). The training countries use different words, so the model had never seen them.

The new features are **computed without labels on each split's own data**, so they transfer to France:
- For each query's top candidate, we take the name tokens present on only one side, and check whether S1's first house number appears in the query's address.
- Per (country, side, token), the smoothed house-number agreement rate measures how often that extra token belongs to the same entity. On train, this rate correlates at 0.98 (US) and 0.99 (India) with the true match rate of the token. Words that mark a different entity ("group", "holdings", "north" in the US) have agreement near 0.02, while spelling and legal-form variants are near 0.7. On test, the same computation flags the French "holding", "participations", "distribution", "international" and "snc" (about 0.03).
- Pair features: the count of extra tokens on each side, the minimum and mean agreement rate, both of these relative to the country's average, and the log-frequency of the most frequent extra token (12 features). `xq_fmax` and `xq_relmin` rank 4th and 5th by gain.

**Features used (64 in v5, 52 in v4, 48 in v2 and v3):**"""),
("- v4 is trained on the fold-B training queries (5.4M pairs, 62% positive).",
 "- v4 and v5 are trained on the fold-B training queries (5.4M pairs, 62% positive)."),
("The chosen threshold is t = 0.75 in v4 (0.65 in v2 and v3).", "The chosen threshold is t = 0.75 in v4 and v5 (0.65 in v2 and v3)."),
("| **v4 + cross-encoder feature** | **0.9885** | 0.9893 | 0.9874 | 0.9983 | 0.9683 | 0.9940 |",
 "| v4 + cross-encoder feature | 0.9885 | 0.9893 | 0.9874 | 0.9983 | 0.9683 | 0.9940 |\n| **v5 + distractor-token statistics** | **0.9890** | 0.9897 | 0.9878 | 0.9985 | 0.9690 | 0.9949 |"),
("Planned improvements (next versions): extra retrieval and features for empty-address queries (name-rarity-aware), France pseudo-labelling, and cross-encoder scores for fold A so the LightGBM can train on all queries.",
"""**Where the remaining validation loss is (v5).** Only 1.7% of true pairs are missing from the candidate set, but adding them back would lift F0.5 from 0.9890 to 0.9944. A perfect model on the existing candidates would reach 0.9947. The two sources are about equal. Of the blocking misses, 56% (India) and 90% (US) have an empty query address. Many have a common name ("Family Health", "Shivam Consulting") with 50 or more same-name S1 entities in the country, where retrieval cannot tell branches apart without an address.

Planned improvements (next versions): cross-encoder scores for fold A so the LightGBM can train on all queries (cross-fitting; in progress), extra retrieval for empty-address queries, and France pseudo-labelling."""),
("It takes macro F0.5 from 0.53 (exact-key baseline) to 0.9885 on held-out entities. The pipeline has no country-specific logic, so the unseen French test data follows the same path. France receives 3.39 matches per S1, in line with 3.38 for US and 3.35 for India.",
 "It takes macro F0.5 from 0.53 (exact-key baseline) to 0.9890 on held-out entities. The pipeline has no country-specific logic, so the unseen French test data follows the same path. Where French vocabulary differs, the model uses statistics computed on the test data itself, without labels. France receives 3.35 matches per S1, in line with 3.38 for US and 3.35 for India."),
("- Records with no address are the main remaining error source.",
 "- A domain shift to an unlabelled country can be partly bridged with self-supervised signals. In this data, house-number agreement indicates which extra name tokens mark a different entity.\n- Records with no address are the main remaining error source."),
("| `src/metric.py` |", "| `src/token_stats.py` | Label-free per-split, per-country statistics of extra name tokens, and the pair features built from them (v5) |\n| `src/metric.py` |"),
("**Threshold sweep (v4, validation):**", """**Threshold sweep (v5, validation):**

| t | F0.5 | Precision | Recall | Singleton accuracy |
|---|---|---|---|---|
| 0.50 | 0.9882 | 0.9963 | 0.9721 | 0.9892 |
| 0.60 | 0.9887 | 0.9975 | 0.9707 | 0.9915 |
| 0.70 | 0.9889 | 0.9982 | 0.9696 | 0.9941 |
| **0.75** | **0.9890** | 0.9985 | 0.9690 | 0.9949 |
| 0.80 | 0.9889 | 0.9988 | 0.9681 | 0.9958 |
| 0.90 | 0.9884 | 0.9993 | 0.9655 | 0.9974 |

**Top features by gain (v5):** `ce_logit`, `p0`, `p0_s1_gap_best`, `xq_fmax`, `xq_relmin`, `ce_qrank`, `q_ncand`, `ce_gap_best`, `num_min_rel`, `ce_gap_2nd`.

**Expected F0.5 on test (Monte-Carlo, p treated as calibrated):**

| Model | France | India | US |
|---|---|---|---|
| v4 | 0.9785 | 0.9944 | 0.9918 |
| v5 | 0.9792 | 0.9950 | 0.9927 |

On validation this estimate is optimistic by 0.003–0.008, so only the difference between versions is meaningful.

**Threshold sweep (v4, validation):**"""),
("**Test prediction statistics (v4):**", """**Test prediction statistics (v5):**
- 13.19M candidate pairs (identical to v2–v4) and 5.82M matches. 99.6% of v4's matched pairs are kept.
- France loses 14.0k v4 matches and gains 4.3k. Most of the dropped pairs are queries whose name adds a distractor word ("Développement", "France", "Snc") or swaps a descriptor ("Club" and "Union").
- 1.633M of 1.733M S1 entities have at least one match. Matches per S1: France 3.35, US 3.38, India 3.35.

**Test prediction statistics (v4):**"""),
("| v4 | 2026-09-25 | 0.9885 | 0.9822 | Adds a fine-tuned e5-small cross-encoder logit and its context as features; LightGBM `lgb_v3` trained on fold B, t = 0.75. Same `candidate_pairs.tsv` as v2 and v3 |",
 "| v4 | 2026-09-25 | 0.9885 | 0.9822 | Adds a fine-tuned e5-small cross-encoder logit and its context as features; LightGBM `lgb_v3` trained on fold B, t = 0.75. Same `candidate_pairs.tsv` as v2 and v3 |\n| v5 | 2026-09-25 | 0.9890 | – | Adds 12 label-free distractor-token features (`src/token_stats.py`); LightGBM `lgb_v4a`, t = 0.75. Same `candidate_pairs.tsv` as v2–v4 |"),
])

patch("business_entity_resolution/README.md", [
("| v4 | `python -m src.run_pipeline --stage all` (`MODEL_NAME=\"lgb_v3\"`, `USE_CE=True`: adds the `ce` stage that fine-tunes and scores the cross-encoder) | 0.9885 |",
 "| v4 | `python -m src.run_pipeline --stage all` (`MODEL_NAME=\"lgb_v3\"`, `USE_CE=True`: adds the `ce` stage that fine-tunes and scores the cross-encoder) | 0.9885 |\n| v5 | `python -m src.run_pipeline --stage all` (`MODEL_NAME=\"lgb_v4a\"`: adds the label-free distractor-token features from `src/token_stats.py`, cached as `work/tokstat_{split}.parquet`) | 0.9890 |"),
])

patch("SUBMISSIONS.md", [
("t=0.75. Same `candidate_pairs.tsv` as v2 and v3 |",
 "t=0.75. Same `candidate_pairs.tsv` as v2 and v3 |\n| v5 | `sub-v5` | 2026-09-25 | 0.9890 (0.9897 / 0.9878) | – | `work/sub5_lgb_v4a/`, `dist/v5/` | Adds 12 label-free distractor-token features (`src/token_stats.py`): per split and country, the house-number agreement rate of name tokens found on only one side, which flags French distractor words such as holding and participations without labels. LightGBM `lgb_v4a` (64 features, fold B), t=0.75. Monte-Carlo test F0.5 for France 0.9785 → 0.9792. Same `candidate_pairs.tsv` as v2–v4 |"),
])
print("docs ok")
