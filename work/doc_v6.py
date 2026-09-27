"""Update Documentation_template.md, README.md and SUBMISSIONS.md for submission v6 (cross-fitted cross-encoder)."""
from pathlib import Path

R = Path(__file__).resolve().parents[1]


def sub(path, pairs):
    p = R / path
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:60], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")


sub("Documentation_template.md", [
    ("**Submission Date:** 2026-09-25 (submission v5)", "**Submission Date:** 2026-09-26 (submission v6)"),
    ("Validation macro F0.5 is **0.9890** (US 0.9897, India 0.9878).",
     "Validation macro F0.5 is **0.9893** (US 0.9900, India 0.9881)."),
    ("- **Leakage control:** the LightGBM is trained only on **fold B** queries, which the cross-encoder never saw. "
     "Validation queries (those touching held-out S1) are excluded from both folds.",
     "- **Leakage control:** the LightGBM is trained only on **fold B** queries, which the cross-encoder never saw. "
     "Validation queries (those touching held-out S1) are excluded from both folds.\n"
     "- **Cross-fitting (v6):** a second cross-encoder is fine-tuned the same way on fold B. Each training query then gets "
     "the logit of the model that did **not** see it, so the LightGBM trains on all training queries (10.8M pairs instead "
     "of 5.4M). Validation and test pairs get the mean of the two models' logits."),
    ("Scoring the 13.2M test pairs takes about 45 minutes.", "Scoring the 13.2M test pairs takes about 45–70 minutes per model."),
    ("**Features used (64 in v5, 52 in v4, 48 in v2 and v3):**", "**Features used (64 in v5 and v6, 52 in v4, 48 in v2 and v3):**"),
    ("- v4 and v5 are trained on the fold-B training queries (5.4M pairs, 62% positive).",
     "- v6 is trained on all training queries (10.8M pairs, 62% positive) with cross-fitted cross-encoder logits. "
     "v4 and v5 were trained on the fold-B training queries only (5.4M pairs)."),
    ("The chosen threshold is t = 0.75 in v4 and v5", "The chosen threshold is t = 0.75 in v4–v6"),
    ("| **v5 + distractor-token statistics** | **0.9890** | 0.9897 | 0.9878 | 0.9985 | 0.9690 | 0.9949 |",
     "| v5 + distractor-token statistics | 0.9890 | 0.9897 | 0.9878 | 0.9985 | 0.9690 | 0.9949 |\n"
     "| **v6 + cross-fitted cross-encoder (LightGBM on all queries)** | **0.9893** | 0.9900 | 0.9881 | 0.9987 | 0.9693 | 0.9958 |"),
    ("Planned improvements (next versions): cross-encoder scores for fold A so the LightGBM can train on all queries "
     "(cross-fitting; in progress), extra retrieval for empty-address queries, and France pseudo-labelling.",
     "**Pseudo-labelling France (tested, not used).** We simulated the France situation with India: a LightGBM trained on "
     "US labels only scores 0.98685 on India validation (t = 0.75). Adding India pairs pseudo-labelled by that model's own "
     "confident predictions (top-1 p ≥ 0.99 and second ≤ 0.01, or all p ≤ 0.01) *lowers* it to 0.98637 (0.98623 with a "
     "0.95 cut-off), while true India labels raise it to 0.98771. Self-training only reinforces what the model already "
     "believes, so v6 does not use it. The simulation also shows that a country without labels loses only about 0.001, so "
     "the lower France estimate comes from French data properties (descriptor-word swaps between entities at the same "
     "address, 13% of French S1 sharing an exact address vs about 5% in US/India) rather than from missing labels.\n\n"
     "Planned improvements (next versions): extra retrieval for empty-address queries and better handling of same-address "
     "sibling entities."),
    ("It takes macro F0.5 from 0.53 (exact-key baseline) to 0.9890 on held-out entities.",
     "It takes macro F0.5 from 0.53 (exact-key baseline) to 0.9893 on held-out entities."),
    ("**Threshold sweep (v5, validation):**",
     "**Threshold sweep (v6, validation):**\n\n"
     "| t | F0.5 | Precision | Recall | Singleton accuracy |\n"
     "|---|---|---|---|---|\n"
     "| 0.50 | 0.9885 | 0.9966 | 0.9724 | 0.9905 |\n"
     "| 0.60 | 0.9891 | 0.9978 | 0.9710 | 0.9929 |\n"
     "| 0.70 | 0.9892 | 0.9984 | 0.9699 | 0.9947 |\n"
     "| **0.75** | **0.9893** | 0.9987 | 0.9693 | 0.9958 |\n"
     "| 0.80 | 0.9892 | 0.9989 | 0.9685 | 0.9962 |\n"
     "| 0.90 | 0.9888 | 0.9993 | 0.9662 | 0.9974 |\n\n"
     "**Top features by gain (v6):** `ce_logit`, `p0`, `p0_s1_gap_best`, `xq_fmax`, `xq_relmin`, `ce_qrank`, `q_ncand`, "
     "`num_min_rel`.\n\n"
     "**Threshold sweep (v5, validation):**"),
    ("| v5 | 0.9792 | 0.9950 | 0.9927 |", "| v5 | 0.9792 | 0.9950 | 0.9927 |\n| v6 | 0.9802 | 0.9953 | 0.9930 |"),
    ("**Test prediction statistics (v5):**",
     "**Test prediction statistics (v6):**\n"
     "- 13.19M candidate pairs (identical to v2–v5) and 5.82M matches. Against v5, 17.5k assignments are dropped, 13.2k "
     "are added and 91 move to a different S1; France accounts for most of the net change (4.3k fewer matches).\n"
     "- 1.633M of 1.733M S1 entities have at least one match.\n\n"
     "**Test prediction statistics (v5):**"),
    ("| v5 | 2026-09-25 | 0.9890 | 0.9836 | Adds 12 label-free distractor-token features (`src/token_stats.py`); "
     "LightGBM `lgb_v4a`, t = 0.75. Same `candidate_pairs.tsv` as v2–v4 |",
     "| v5 | 2026-09-25 | 0.9890 | 0.9836 | Adds 12 label-free distractor-token features (`src/token_stats.py`); "
     "LightGBM `lgb_v4a`, t = 0.75. Same `candidate_pairs.tsv` as v2–v4 |\n"
     "| v6 | 2026-09-26 | 0.9893 | – | Cross-fitted cross-encoder (second e5-small trained on fold B); LightGBM "
     "`lgb_v5cf` trained on all training queries, t = 0.75. Same `candidate_pairs.tsv` as v2–v5 |"),
])

sub("business_entity_resolution/README.md", [
    ("- **Cross-encoder with a fold split.** The cross-encoder is fine-tuned on fold A of the training queries (a hash half). "
     "The LightGBM that uses its score trains on fold B, so it never sees scores for pairs the cross-encoder was trained on.",
     "- **Cross-fitted cross-encoder.** Two cross-encoders are fine-tuned, one on each hash half (fold A / fold B) of the "
     "training queries. Each training query gets the logit of the model that did not see it, so the LightGBM trains on all "
     "queries without leakage. Validation and test pairs get the mean of both logits (`CROSS_FIT=True`, v6)."),
    ("| `src/cross_encoder.py` | e5-small fine-tuned as a pair classifier (fold A); cached `ce_logit` for train fold B, validation and test |",
     "| `src/cross_encoder.py` | e5-small fine-tuned as a pair classifier (one model per fold); cached out-of-fold `ce_logit` for train, validation and test |"),
    ("| ce (fine-tune ~35 min + score ~70 min) | ~1 h 45 min |", "| ce (2 folds × (fine-tune ~35 min + score ~70 min)) | ~3 h 30 min |"),
    ("| v5 | `python -m src.run_pipeline --stage all` (`MODEL_NAME=\"lgb_v4a\"`: adds the label-free distractor-token "
     "features from `src/token_stats.py`, cached as `work/tokstat_{split}.parquet`) | 0.9890 |",
     "| v5 | `python -m src.run_pipeline --stage all` (`MODEL_NAME=\"lgb_v4a\"`, `CROSS_FIT=False`: adds the label-free "
     "distractor-token features from `src/token_stats.py`, cached as `work/tokstat_{split}.parquet`) | 0.9890 |\n"
     "| v6 | `python -m src.run_pipeline --stage all` (`MODEL_NAME=\"lgb_v5cf\"`, `CROSS_FIT=True`: trains the fold-B "
     "cross-encoder too, and the LightGBM on all training queries) | 0.9893 |"),
])

sub("SUBMISSIONS.md", [
    ("Same `candidate_pairs.tsv` as v2–v4 |\n",
     "Same `candidate_pairs.tsv` as v2–v4 |\n"
     "| v6 | `sub-v6` | 2026-09-26 | 0.9893 (0.9900 / 0.9881) | – | `work/sub6_lgb_v5cf/`, `dist/v6/` | Cross-fitting: a "
     "second e5-small cross-encoder fine-tuned on fold B, so every training query gets an out-of-fold `ce_logit` (validation "
     "and test: mean of both). LightGBM `lgb_v5cf` (64 features) trained on all 10.8M training pairs instead of fold B only, "
     "t=0.75. Monte-Carlo test F0.5 for France 0.9792 → 0.9802. Pseudo-labelling France was tested and rejected (India "
     "simulation: −0.0005). Same `candidate_pairs.tsv` as v2–v5 |\n"),
])
print("ok")
