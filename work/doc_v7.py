"""Update Documentation_template.md, README.md and SUBMISSIONS.md for submission v7 (context-keyed token statistics)."""
from pathlib import Path

R = Path(__file__).resolve().parents[1]

def sub(path, pairs):
    p = R / path
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:60], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")

CTX = ("- **Context-keyed statistics (v7).** The same word can mark a different entity in one context and be harmless noise "
       "in another. In French test data, `groupe` *added* in the middle of a name (the other side has no extra tokens) has a "
       "relative agreement of 0.02, while `groupe` *swapped* for another suffix word is at 1.3. The v5 per-token rate averages "
       "the two (about 0.5) and scores both badly. v7 adds a second table keyed by (country, side, token, kind, position): kind "
       "is `a` (added) or `s` (swapped), position is first, last or middle of the name. It is built the same label-free way. "
       "On train, the AUC of the relative rate against the true label rises from 0.834 to 0.850 for query-side tokens and from "
       "0.611 to 0.729 for S1-side tokens. Six pair features: minimum and mean relative rate, and the minimum log-frequency of "
       "the context key, per side (`xq_krelmin`, `xq_krelmean`, `xq_kfmin`, `x1_*`). `xq_krelmin` becomes the 4th feature by gain.\n")

sub("Documentation_template.md", [
    ("**Submission Date:** 2026-09-26 (submission v6)", "**Submission Date:** 2026-09-26 (submission v7)"),
    ("and the log-frequency of the most frequent extra token (12 features). `xq_fmax` and `xq_relmin` rank 4th and 5th by gain.\n",
     "and the log-frequency of the most frequent extra token (12 features). `xq_fmax` and `xq_relmin` rank 4th and 5th by gain.\n" + CTX),
    ("**Features used (64 in v5 and v6, 52 in v4, 48 in v2 and v3):**", "**Features used (70 in v7, 64 in v5 and v6, 52 in v4, 48 in v2 and v3):**"),
    ("| **v6 + cross-fitted cross-encoder (LightGBM on all queries)** | **0.9893** | 0.9900 | 0.9881 | 0.9987 | 0.9693 | 0.9958 |",
     "| v6 + cross-fitted cross-encoder (LightGBM on all queries) | 0.9893 | 0.9900 | 0.9881 | 0.9987 | 0.9693 | 0.9958 |\n"
     "| **v7 + context-keyed token statistics (t = 0.70)** | **0.9893** | 0.9900 | 0.9881 | 0.9984 | 0.9699 | 0.9949 |"),
    ("**Top features by gain (v6):**",
     "**v7 (validation):** t = 0.70 is chosen (F0.5 0.98927; 0.9892–0.9893 for t between 0.60 and 0.80). The validation score "
     "equals v6: the context features mainly help with vocabulary that is rare in US and India. On test, v7 adds 20.2k "
     "assignments and drops 6.0k compared with v6; 12.1k of the additions are French, and 85% of those have agreeing house "
     "numbers (v6's French assignments: 98%). The simulation with doubled test distractor density keeps t between 0.70 and 0.80 "
     "optimal (0.98895–0.98898).\n\n"
     "**Top features by gain (v7):** `ce_logit`, `p0`, `ce_qrank`, `xq_krelmin`, `xq_fmax`, `ce_gap_best`, `num_min_rel`, "
     "`q_ncand`, `ce_gap_2nd`, `xq_n`, `xq_relmin`.\n\n"
     "**Top features by gain (v6):**"),
    ("| v6 | 0.9802 | 0.9953 | 0.9930 |", "| v6 | 0.9802 | 0.9953 | 0.9930 |\n| v7 | 0.9805 | 0.9953 | 0.9930 |"),
    ("LightGBM `lgb_v5cf` trained on all training queries, t = 0.75. Same `candidate_pairs.tsv` as v2–v5 |",
     "LightGBM `lgb_v5cf` trained on all training queries, t = 0.75. Same `candidate_pairs.tsv` as v2–v5 |\n"
     "| v7 | 2026-09-26 | 0.9893 | – | Adds 6 context-keyed (token, added/swapped, position) label-free token features; "
     "LightGBM `lgb_v6k` (70 features), t = 0.70. Same `candidate_pairs.tsv` as v2–v6 |"),
])

sub("business_entity_resolution/README.md", [
    ("trains the fold-B cross-encoder too, and the LightGBM on all training queries) | 0.9893 |",
     "trains the fold-B cross-encoder too, and the LightGBM on all training queries) | 0.9893 |\n"
     "| v7 | `python -m src.run_pipeline --stage all` (`MODEL_NAME=\"lgb_v6k\"`: adds context-keyed token statistics, cached as "
     "`work/tokstat2_{split}.parquet`; threshold 0.70) | 0.9893 |"),
])

sub("SUBMISSIONS.md", [
    ("Same `candidate_pairs.tsv` as v2–v5 |\n",
     "Same `candidate_pairs.tsv` as v2–v5 |\n"
     "| v7 | `sub-v7` | 2026-09-26 | 0.9893 (0.9900 / 0.9881) | – | `work/sub7_lgb_v6k/`, `dist/v7/` | Label-free token "
     "statistics keyed also by context: added vs swapped token and its position in the name (first, last, middle). This "
     "separates, for example, French `groupe` added mid-name (a distractor marker, relative agreement 0.02) from `groupe` "
     "swapped at the end (noise, 1.3). 6 new features, LightGBM `lgb_v6k` (70 features), t=0.70. Validation flat; Monte-Carlo "
     "test F0.5 for France 0.9802 → 0.9805; +12.1k / −3.8k French assignments vs v6. Same `candidate_pairs.tsv` as v2–v6 |\n"),
])
print("ok")
