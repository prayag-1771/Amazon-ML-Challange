## Submission v15: India addressed no-candidate channel (val F0.5 0.9906)

**Upload `matching_results.tsv` (Madhav).**

### Changes vs v14
- New `src/india_addr.py`. It targets India queries that have an address but no pair in the cascade candidate set, excluding any the empty-address rescue already covers. These are mostly transliterated names ("Shri Ganesh" / "Sree Ganesha") whose address overlaps the true S1 only partly.
- Retrieval, per state:
  - The top 30 S1 by address-word TF-IDF.
  - The top 30 by a phonetic name skeleton: vowels and h dropped, ph→f, c→k/s, z→s, w→v, d→t, b→p, g/j→k, m→n, repeats squeezed, then char_wb 2–3-gram TF-IDF.
  - Kept: the combined top 10, plus the top 2 of each channel.
- Scoring:
  - A 33-feature LightGBM over 3 seeds (31 leaves, min 300 per leaf, L2 5).
  - Trained on training queries whose true pair blocking missed, excluding queries that touch a validation S1.
  - The top-1 pair is accepted at ≥ 0.9.
- Accepted pairs go into both TSVs. Stage 2, the rules and the rescue are unchanged from v14.
- Ranking ties are now broken in a fixed order, so every run gives the same output.

### Validation
| | F0.5 | US | India | Precision | Recall |
|---|---|---|---|---|---|
| v14 | 0.99019 | 0.99087 | 0.98917 | 0.99896 | 0.97132 |
| **v15** | **0.99055** | 0.99087 | 0.99007 | 0.99895 | 0.97224 |

The channel adds 713 validation pairs at precision 0.989. Threshold sweep: 0.8 gives +0.00043 (precision 0.962), 0.9 gives +0.00036 (precision 0.989), 0.97 gives +0.00022. We use 0.9 for safety, because test density differs.

### Test-side
- Matches: 5,824,560 (+5,505 India, none removed vs v14). Each added query is India, maps to one S1 and has no cascade or rescue pair.
- Candidates: 6,918,451 (3.99 per S1).
- Validator: PASS.

### Leaderboard history
v4 0.9822 · v5 0.9836 · v6 0.9841 · v7 0.9844 · v9 0.9848 · probe11 0.9879 · v13 0.98824
