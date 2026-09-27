**Validation macro F0.5: 0.9895** (US 0.9903 / India 0.9883; 10% held-out S1 entities). v8 was 0.9893, so the gain is **+0.0002**, which is small. Public LB history: v6 = **0.9841**, v7 = **0.9844** (rank 69).

**What changed:** a stage-2 collective re-ranker (`src/stage2.py`).
- Input: the v8 ensemble scores (mean of `lgb_v5cf` and `lgb_v6k`) for every candidate pair.
- New features look beyond a single pair:
  - How many other S2/S3 records confidently pick the same S1 (p ≥ 0.9 / 0.5), split into same source and the other source.
  - The pair's rank among that S1's candidates.
  - The same support counts for the competing S1 in the query.
  - The query's own score gap.
- A second LightGBM is trained on the validation queries, because first-stage scores are out-of-sample there.
- Quality was checked with 5-fold CV grouped by top-1 S1: 0.98930 → 0.98951.
- The gain is the same when the share of unmatched queries is simulated at test-like levels (40–58%).
- Thresholds: US 0.75, India 0.80, France 0.75.

Total matched pairs: 5,821,273 (US 2,242,074; India 2,711,434; France 867,765).

`candidate_pairs.tsv` is identical to v2–v8.

## Files
| File | Use |
|---|---|
| `matching_results.tsv` | **Upload this to the leaderboard.** It is the only file that is scored. |
| `candidate_pairs.tsv` | Candidate set the model scored. It goes inside the final zip and is not uploaded on its own. |
| `Greedy_Decoders_submission.zip` | Final package: output, code and filled-in documentation. Submit it once, at the end, for the version the team picks. |

## SHA-256
```
e61d8b59202187d520ef398f4b0e42ee3f82f071de037148daa9a68932f9068b  output/matching_results.tsv
9a85964c7d870a44232ebdd59be907e76f5a209f4ee012ee9f1c7734e88e606c  output/candidate_pairs.tsv
bab8294104c04b0aa7a5333b19c21c5bd087c01818dac411a82d7171c74a2d30  Greedy_Decoders_submission.zip
```
Both TSVs pass `student_resource/utils/validate_submission.py --check-ids`.
