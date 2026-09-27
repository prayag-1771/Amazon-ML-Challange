## Submission v13: empty-address name rescue (val F0.5 0.9901)

**Upload `matching_results.tsv` (Madhav).**

### Changes vs v12
- **Empty-address name rescue** (`src/rescue.py`).
  - Queries with no address often get no candidate from address-heavy blocking.
  - For empty-address queries with no pair in the v12 cascade set, a name-only char_wb 3-gram TF-IDF (per country, US/India) retrieves the top 10 S1.
  - A 20-feature LightGBM scores those pairs. Its features cover name similarity, gaps to the best and second-best candidate, ties, same-name S1 and query counts, and rapidfuzz scores.
  - The top-1 pair is accepted when its score is ≥ 0.9.
  - The model is trained on training queries that touch no validation S1.
- Rescued pairs are appended to both `matching_results.tsv` and `candidate_pairs.tsv`, so matches remain a subset of candidates.
- All other v12 matches are unchanged: 0 removed.

### Validation (10% held-out S1)
| | F0.5 | US | India | Precision | Recall |
|---|---|---|---|---|---|
| v12 | 0.98989 | 0.99066 | 0.98874 | 0.99902 | 0.97029 |
| **v13** | **0.99009** | 0.99071 | 0.98916 | 0.99894 | 0.97106 |

The rescue adds 643 validation pairs at precision 0.914.

### Test-side
- Matches: 5,817,564, which is v12 plus 3,256 rescued pairs (India 3,045, US 211).
- Candidates: 6,912,946 (3.99 per S1).
- Validator: PASS.

### Leaderboard history
v4 0.9822 · v5 0.9836 · v6 0.9841 · v7 0.9844 · v9 0.9848 · v11 + France vocab probe 0.9879
