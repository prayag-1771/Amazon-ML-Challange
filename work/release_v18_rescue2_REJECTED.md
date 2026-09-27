## v18 candidate: empty-address rescue v2. **REJECTED (negative once measured correctly)**

### What was tried
- `work/v18/rescue_v2.py` trains a LightGBM on the candidate pairs from `work/v10/resc2.py`: three name-only retrieval views per country, 8.5M train pairs, 344k empty-address records.
- For every empty-address record that v15 leaves unassigned, it takes the top-1 pair if it clears a cutoff.
- **Set A** uses the resc2 features. **Set B** adds tie features: the S1's addressed-match count, its gap to the least-matched same-name S1 in the record's candidates, the tie-group size, and whether the legal form singles out one S1 of the group.

### The trap: how validation records are chosen
- **Original resc2 split.** It held out every record with *any* validation-S1 candidate. That is unbiased, but with ~25 candidates per record only ~7% of the data is left for training. Result: every cutoff lowers F (set A: −0.00004 to −0.0009).
- **"True S1" split.** Validation = records whose true S1 is a validation S1. Training = the rest, without validation-S1 rows. This trains on 6.8M pairs, and the naive validation looks good:

| Set | Best cutoff | Naive gain | Pairs added | Precision |
|---|---|---|---|---|
| A | 0.3 | **+0.00042** | 2,919 | 0.84 |
| B | 0.3 | **+0.00060** | 3,285 | 0.86 |

  B is still +0.00058 with the addressed-match count taken from predictions, as it would be on test.
- **Why that is misleading.** Every correct pick of these records lands on a validation S1 and is counted. But ~90% of their wrong picks land on non-validation S1 and are invisible to the validation score. On test, the other 90% of records produce as many wrong picks landing on validation S1. So the real number of false merges on validation S1 is about *all* wrong picks of the validation records.
  - Unmatched records are the exception: almost all of them touch a validation S1, so they are already fully counted.

### Corrected result (`v18/eval_split.py`)

| Cutoff | Precision, all picks (B) | Corrected gain A | Corrected gain B |
|---|---|---|---|
| 0.3 | 0.48 | −0.00260 | −0.00251 |
| 0.5 | 0.57 | −0.00108 | −0.00080 |
| 0.7 | 0.64 | −0.00021 | −0.00020 |
| 0.9 | 0.71 | −0.00007 | −0.00006 |
| 0.95 | 0.72 | −0.00003 | −0.00003 |

- A false merge costs about 1.2e-6 of validation F; a new true match gains about 4.1e-7. So added pairs need **≥ ~75% precision** to pay off.
- The records v15 leaves unassigned never reach that. Most are genuine same-name ties, plus decoy records, which make up a third of the unassigned pool.

### Decision
Do not ship rescue v2. This also answers the plan in the team notes ("strip generic words, add word matching, retrieve more, retrain"): with honest false-merge counting, it does not pay off.

### Lesson for any new channel
Validate on *all* records that could land on a validation S1, not only on records whose true S1 is a validation S1. If that leaves too little training data, apply the correction in `v18/eval_split.py`.
