## Submission v14: exact house-number shift features in stage 2 (val F0.5 0.9902)

**Upload `matching_results.tsv` (Madhav).**

### Changes vs v13
- Stage 2 gets three new features:
  - `ndiff`: the exact signed first-house-number shift, capped at ±30 and folded to ±99 beyond.
  - `nd3`: a flag for the distractor generator's offsets (3, 4, 5, 7, 9, 11, 13, 21).
  - `nratio`: the number ratio.
- v10's `nrel_c` bucketed every shift into ±1..25, so an offset of 3 (distractor) looked the same as 6 (mostly a true match).
- The empty-address rescue (v13) is unchanged.

### Validation
| | F0.5 | US | India | Precision | Recall |
|---|---|---|---|---|---|
| v13 | 0.99009 | 0.99071 | 0.98916 | 0.99894 | 0.97106 |
| **v14** | **0.99019** | 0.99087 | 0.98917 | 0.99896 | 0.97132 |

Single-model 5-fold CV gained +0.0001 on each of 3 seeds (0.98980 → 0.98991 / 0.98981 → 0.98992 / 0.98980 → 0.98993).

### Test-side
- Matches: 5,819,055 (France 858,237, India 2,716,341, US 2,244,477).
- Against v13: +3,950 / −2,459 pairs.
- Estimated excess up-shifted acceptances (label-free false-positive estimate): US 925 → 759; France and India stay ≤ 0.
- Candidates: 6.91M (3.99 per S1), unchanged.
- Validator: PASS.

### Leaderboard history
v4 0.9822 · v5 0.9836 · v6 0.9841 · v7 0.9844 · v9 0.9848 · probe11 0.9879 · v13 0.98824
