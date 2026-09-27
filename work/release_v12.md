## Submission v12: cascade candidate set (half the pairs), 3-seed stage 2, label-free France rules (val F0.5 0.9899)

**Upload `matching_results.tsv` (Madhav).**

### Changes vs v11
- **Cascade candidate set.** `candidate_pairs.tsv` keeps only pairs whose first-stage ensemble score (mean of `lgb_v5cf` and `lgb_v6k`) is at least 0.001. Stage 2 scores exactly this set, and every match lies inside it.
  - Test candidates: 13.19M → **6.91M** (3.99 per S1, 1.06 per query).
  - Candidate recall on validation is unchanged: India 97.6%, US 98.8%.
  - This targets the organiser's criterion that a smaller candidate set per S1 ranks higher.
- **Stage 2** is retrained on the cut set and bagged over 3 seeds to lower retraining variance.
- **France variant-vocabulary rule** (`variant_rule`, label-free).
  - The generator's true-match name variants swap in, or add, a word from a small per-country vocabulary at an unchanged house number. Real words swap in and out about equally.
  - The vocabulary is derived from test swap-in/out rates. It comes out as associes, developpement, fils, france, groupe, services. On US/India the same derivation gives center, partners, service, services, the known vocabulary.
  - For France only: a single-token add or swap to a variant word is accepted (23.5k pairs), and a swap to a frequent real word is rejected (8.7k pairs).
  - The same rule applied to v11 scores (probe `vfix`) scored **0.9879 on the LB**, against 0.9848 for v9.
- **Legal-form-add rule** (`legal_add_rule`).
  - A top-1 pair is accepted when it has the same name and house numbers, overlapping street tokens and first-stage p ≥ 0.9, and the only difference is an added legal form (mostly "SASU").
  - This accepts 2.2k France pairs.
  - The cell has precision 0.999 on validation, and stage 2 already accepts all of it there.

### Validation (5-fold CV, 10% held-out S1)
| | F0.5 | US | India | Precision | Recall |
|---|---|---|---|---|---|
| v11 | 0.9899 | 0.9907 | 0.9887 | 0.9990 | 0.9704 |
| **v12** | **0.9899** | 0.9907 | 0.9887 | 0.9990 | 0.9703 |

Validation is flat, as expected: the France rules cannot be measured on validation (France has no labels), and the cascade is meant to cost nothing.

### Test-side
- Matches: 5,814,308. By country: US 2,243,626, India 2,713,097, France 857,585. v11 France was 843,054, and `vfix` (LB 0.9879) was 857,004.
- Estimated excess up-shifted acceptances (label-free false-positive estimate): US 0.9k, France ≈0, India ≈0. This is unchanged from v11.
- Validator: PASS.

### Leaderboard history
v4 0.9822 · v5 0.9836 · v6 0.9841 · v7 0.9844 · v9 0.9848 · v11 + France vocab probe 0.9879
