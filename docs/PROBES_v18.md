# v18 leaderboard probes: France cutoff (plan fixed before any upload)

**Context (2026-09-27, evening).** The best leaderboard (LB) score is 0.98873, and 4 uploads are left. The final version v17 (`work/sub17_final/`) and the final zip are **not changed** by anything here. Probe files live in `work/v18p/<name>/`.

## Why France, and why its cutoff
- France has no labels, so France changes can only be measured on the LB.
- Estimated France F0.5 is about 0.979, against about 0.990 for US/India.
- Per S1, France has **3–5× more** top-1 records in the rejected-but-uncertain band (stage-2 p2 in 0.3–0.75): 0.134 per S1, vs 0.040 (US) and 0.028 (India).
- France also accepts fewer pairs per S1: 3.31, vs 3.39 (US) and 3.37 (India).
- Both point to France *under*-accepting, but that can't be confirmed without labels.

France top-1 records by stage-2 score band (raw p2, before rules):

| Band | 0.3–0.5 | 0.5–0.6 | 0.6–0.75 | 0.75–0.85 | 0.85–0.9 | 0.9–0.95 |
|---|---|---|---|---|---|---|
| France | 17,210 | 7,003 | 10,403 | 9,102 | 6,743 | 11,363 |

**Expected effect.** A new true match gains about 5e-8 LB and a new false merge costs about 1.45e-7, so added pairs break even at about 74% precision. Moving the France cutoff by one band is therefore worth about ±0.0002–0.0004 LB.

## Probe files
Each is v17 with only `T["France"]` changed. US and India are identical to v17, and so is the candidate file. Built by `work/v18p/build_probe.py`, which reproduces v17 byte for byte at cutoff 0.75.

| Folder | France cutoff | France matches | Total matches | Validator |
|---|---|---|---|---|
| `work/sub17_final/` (v17) | 0.75 | 859,043 | 5,828,779 | PASS |
| `work/v18p/fr060/` | 0.60 | 864,018 (+4,975) | 5,833,754 | PASS |
| `work/v18p/fr050/` | 0.50 | 867,737 (+8,694) | 5,837,473 | PASS |
| `work/v18p/fr085/` | 0.85 | 854,795 (−4,248) | 5,824,531 | PASS |

The sha256 of each file is in its folder's `MANIFEST.txt`. The number-missing rule adds 607 / 495 / 1,024 pairs at 0.60 / 0.50 / 0.85, because a lower cutoff already accepts some of them directly.

## Upload plan (fixed in advance, so the public LB is not overfitted)
1. **Upload v17** (`work/sub17_final/matching_results.tsv`) → score S17. This is the reference.
2. **Upload fr060** → score S60, and compute Δ = S60 − S17.
   - Δ > +0.00005 means France is under-accepting. **Upload fr050** next.
   - Δ < −0.00005 means France is over-accepting. **Upload fr085** next.
   - |Δ| ≤ 0.00005 means no measurable effect. **Stop:** v17 stays final and the remaining uploads are kept.
3. **Final choice** = the highest-scoring file among those already uploaded. No extrapolation, and no fine search on the public LB.
4. The 4th upload is kept in reserve (in case the portal ranks by the last upload rather than the best).

**Anti-overfitting.** Only one parameter is tested, with at most two probes. Each probe changes about 10–17k France pairs, far above leaderboard noise. US and India are never touched.

## Results (fill in after each upload)
| Upload | File | Public LB | Δ vs v17 |
|---|---|---|---|
| 1 | v17 | **0.9888** (previous best 0.98873) | – |
| 2 | fr060 | | |
| 3 | | | |
