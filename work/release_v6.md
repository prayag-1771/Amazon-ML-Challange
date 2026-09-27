**Validation macro F0.5: 0.9893** (US 0.9900 / India 0.9881; 10% held-out S1 entities). v5 scored 0.9890 on validation and **0.9836** on the public leaderboard.

**What changed.** Cross-fitting of the cross-encoder.
- A second e5-small cross-encoder is fine-tuned on fold B, the half of the training queries the first one never saw.
- Every training query now gets an out-of-fold `ce_logit`. The LightGBM (`lgb_v5cf`, 64 features, t = 0.75) therefore trains on all 10.8M training pairs instead of the 5.4M in fold B.
- Validation and test pairs get the mean of the two cross-encoders' logits.

Results:
- Micro precision: 0.9987 (v5: 0.9985).
- Micro recall: 0.9693 (v5: 0.9690).
- Singleton accuracy: 0.9958 (v5: 0.9949).
- Monte-Carlo test estimate: France 0.9792 → 0.9802, India 0.9950 → 0.9953, US 0.9927 → 0.9930.

**Tested and not used: pseudo-labelling France.** We simulated France with India.
- A model trained on US labels only scores 0.98685 on India validation.
- Adding India pairs labelled by that model's own confident predictions lowers the score to 0.98637.
- Adding real India labels raises it to 0.98771.

`candidate_pairs.tsv` is identical to v2–v5. Against v5, 17.5k assignments are dropped, 13.2k are added and 91 move to a different S1.

## Files
| File | Use |
|---|---|
| `matching_results.tsv` | **Upload this to the leaderboard.** It is the only file that is scored. |
| `candidate_pairs.tsv` | Candidate set the model scored. It goes inside the final zip and is not uploaded on its own. |
| `Greedy_Decoders_submission.zip` | Final package: output, code and filled-in documentation. Submit it once, at the end, for the version the team picks. |

## SHA-256
```
075dd7e3470b3dafe87eca6f07b0c20b07cf7329d9920adfe70a5081ae5b3c13  output/matching_results.tsv
9a85964c7d870a44232ebdd59be907e76f5a209f4ee012ee9f1c7734e88e606c  output/candidate_pairs.tsv
363e33e89343567f28c063f2d224caf545ecab9ae346b3c5b685cefbc6e31250  Greedy_Decoders_submission.zip
```
Both TSVs pass `student_resource/utils/validate_submission.py --check-ids`.
