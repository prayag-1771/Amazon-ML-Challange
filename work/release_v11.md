## Submission v11: hard-pair cross-encoder, shift-symmetry cap (val F0.5 0.9899)

**Upload `matching_results.tsv` (Madhav).**

### Changes vs v10
- **v10 likely hurts on test.** Its sibling features `sib_num` / `sib_name` count other candidate queries of the same S1 that share the house number or name, and they flipped meaning on test. The test set contains a distractor family absent from validation: legal form added and house number shifted up by 3–21. Its members share the shifted number, so the feature marked them as agreeing. In US cell (shift in {3,4,5,7,9,11,13,21}, legal form added, same name), acceptances went from 153 in v9 to 18,985 in v10, against about 217 true pairs estimated from the down-shifted pairs.
- **Removed** `sib_num` and `sib_name` from stage 2.
- **Added** the hard-pair cross-encoder score (`src/ce_hard.py`). It is an e5-small model fine-tuned on the ambiguous band of training pairs and adds three features: `ce_h`, its rank and its gap to the best score within the query. It adds +0.0003 CV and mostly rejects generic-word swaps, which are the main France error.
- **Shift-symmetry cap** (`shift_cap`, label-free). True matches drift up and down in house number about equally, while distractors only drift up. In each (country, shift bucket, legal-form relation, same name) cell, accepted up-shifted pairs are capped at the level the accepted down-shifted pairs imply, with generous slack. It never fires on validation; on test it rejects 2,977 France pairs.

### Validation (5-fold CV, 10% held-out S1)
| | F0.5 | US | India | Precision | Recall |
|---|---|---|---|---|---|
| v9 | 0.9895 | 0.9903 | 0.9883 | 0.9989 | 0.9695 |
| v10 | 0.9898 | 0.9908 | 0.9884 | 0.9990 | 0.9705 |
| **v11** | **0.9899** | 0.9907 | 0.9887 | 0.9990 | 0.9704 |

### Test-side label-free check (estimated excess up-shifted acceptances ≈ false positives)
| | US | France | India |
|---|---|---|---|
| v9 | 1.7k | 3.8k | 0.6k |
| v10 | 30.1k | 10.1k | 1.8k |
| **v11** | **0.9k** | **≈0** | **≈0** |

France accepted pairs per S1: 3.33 without the ce_h feature, 3.25 with it.

Matches: 5,800,714 (US 2,244,267; India 2,713,393; France 843,054). The candidate pairs are unchanged from v10 (about 1.27 per query). Validator: PASS.

### Leaderboard history
v4 0.9822 · v5 0.9836 · v6 0.9841 · v7 0.9844 · v9 0.9848 (rank 117)
