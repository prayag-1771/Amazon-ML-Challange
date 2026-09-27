**Validation macro F0.5: 0.9898** (US 0.9908 / India 0.9884; 10% held-out S1 entities, stage-2 5-fold CV). v9 was 0.9895, so the gain is **+0.0003**. Public LB history: v6 = **0.9841**, v7 = **0.9844** (rank 69), v9 = **0.9848** (0.984799, rank 117).

**What changed:** 9 new features in the stage-2 re-ranker (`src/stage2.py`).
- Distractors repeat an S1 name, add or swap a generic token, and shift the house number by 1–25.
- Structure features:
  - How the first house numbers relate: equal, +1..25, −1..25, far or missing.
  - How many name tokens the query adds, and how many it drops.
- Sibling-agreement features. Among the other candidate queries of the same S1, count how many:
  - share this query's first house number;
  - share its exact name;
  - contain its added tokens (min and max over those tokens);
  - lack its dropped tokens (min and max over those tokens).

  A distractor family repeats the same deviation, while a true record's deviation is unique.
- 5-fold CV grouped by top-1 S1: 0.98950 → 0.98984. When trained on US and scored on India, the result is 0.98825 → 0.98835, so the gain carries over to an unseen country.
- Thresholds are unchanged: US 0.75, India 0.80, France 0.75.

Total matched pairs: 5,867,511 (US 2,278,429; India 2,715,711; France 873,371).

`candidate_pairs.tsv` is identical to v2–v9.

## Files
| File | Use |
|---|---|
| `matching_results.tsv` | **Upload this to the leaderboard.** It is the only file that is scored. |
| `candidate_pairs.tsv` | Candidate set the model scored. It goes inside the final zip and is not uploaded on its own. |
| `Greedy_Decoders_submission.zip` | Final package: output, code and filled-in documentation. Submit it once, at the end, for the version the team picks. |

## SHA-256
```
039c6dae4364765893be711291c6af4a31a8ca5c30b96b50455ca18d935f86ca  output/matching_results.tsv
9a85964c7d870a44232ebdd59be907e76f5a209f4ee012ee9f1c7734e88e606c  output/candidate_pairs.tsv
37314f6507825061a95011e5a40538f1893db8993617725d904007e7d4c471d4  Greedy_Decoders_submission.zip
```
Both TSVs pass `student_resource/utils/validate_submission.py --check-ids`.
