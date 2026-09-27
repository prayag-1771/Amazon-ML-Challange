**Validation macro F0.5: 0.9893** (US 0.9901 / India 0.9881; 10% held-out S1 entities). Public LB history: v6 = **0.9841**, v7 = **0.9844** (rank 69).

**What changed:** Model Ensemble (v5cf + v6k) with Per-Country Calibrated Decision Thresholding.
- Evaluates an ensemble of two high-performing cross-fitted LightGBM models: `lgb_v5cf` (64 features, trained across all 10.8M training pairs) and `lgb_v6k` (70 features, incorporating context-keyed token statistics).
- Equal-weight blend of the two models' probabilities. On validation this is essentially flat versus v7: 0.98927 to 0.98930 (US 0.99004 to 0.99007, India 0.98811 to 0.98815). Low-risk variance reduction, not a real step up.
- Calibration analysis revealed that India suffers from precision degradation at lower thresholds, whereas US and France benefit from moderate thresholds. Per-country decision thresholding is deployed:
  - **US:** $t = 0.70$ (maximizes validation F0.5 at 0.99007 with 0.9721 recall and 0.9984 precision).
  - **India:** $t = 0.80$ (maximizes validation F0.5 at 0.98815, boosting micro precision to 0.99901 while preserving true matches).
  - **France:** $t = 0.70$ (maximizes Monte-Carlo expected F0.5 at 0.97990, yielding 870,564 matched pairs, exactly 3.355 pairs per S1 consistent with ground-truth entity density).

Total matched pairs: 5,823,223 (US: 2,242,176; India: 2,710,483; France: 870,564).

`candidate_pairs.tsv` is identical to v2–v7.

## Files
| File | Use |
|---|---|
| `matching_results.tsv` | **Upload this to the leaderboard.** It is the only file that is scored. |
| `candidate_pairs.tsv` | Candidate set the model scored. It goes inside the final zip and is not uploaded on its own. |
| `Greedy_Decoders_submission.zip` | Final package: output, code and filled-in documentation. Submit it once, at the end, for the version the team picks. |

## SHA-256
```
9fce4f803d31178d4c07ac7b37774c467787252cebb870eccccf3b89678b283c  output/matching_results.tsv
9a85964c7d870a44232ebdd59be907e76f5a209f4ee012ee9f1c7734e88e606c  output/candidate_pairs.tsv
4c70ce07af6a83b89fd8cf3d9189cfd1d3ef29179e821362c020daa6e665fc48  Greedy_Decoders_submission.zip
```
Both TSVs pass `student_resource/utils/validate_submission.py --check-ids`.
