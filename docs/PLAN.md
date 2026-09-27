# Improvement plan

Last updated 2026-09-27. Gains are estimates on the public leaderboard (LB) unless marked "val".

## Where the score can and cannot go

- **The LB ceiling is about 0.9967.** Empty-address records are about 97.7% true matches, but when their name is shared by two or more Source-1 businesses nothing identifies the owner. Four ways of breaking the tie were tested; none helped. This costs about 0.0033 on the LB and 0.0057 on validation. Details: [analysis/2026-09-27-loss-analysis.md](analysis/2026-09-27-loss-analysis.md).
- **Realistic target: about 0.989–0.990**, from about 0.9886 (v15, estimated). 0.999 is not reachable.

## Steps

| # | Step | Owner | Est. gain | Status |
|---|---|---|---|---|
| 1 | **v16**: India addressed no-candidate channel with a third retrieval method and retrained models. Recommended cutoff **0.7** rather than 0.6: val +0.00013 at precision 0.949, versus +0.00015 at 0.931. The test set has about 50% more decoy records than validation. | peer session (other machine) | +0.0001–0.0002 | In progress there; not on this machine |
| 2 | **v17**: France number-missing rule, in `work/v17/` | this repo (2026-09-27) | +0.00004 | **Done.** Built and validated (`work/sub17_fr_on_v15/`). Too small for its own upload slot, so bundle it with the next change. Apply to any base with `python v17/build.py <base_dir> <out_dir> v17/fr_patch.parquet`. |
| 3 | Empty-address rescue v2 (`work/v18/rescue_v2.py`) | this repo (2026-09-27) | **negative** | **Rejected.** The naive validation gain is +0.0006, but with false merges counted properly it is −0.00003 to −0.0025 at every cutoff. See `work/release_v18_rescue2_REJECTED.md`. |
| 4 | Stage-2 features for empty-address name ties | open | ~0 | **Deprioritised.** The same tie features could not lift rescue precision above 0.72, and a false merge costs about 3× a true match (break-even about 0.75). |
| 5 | Stage-2 cutoff under test-like decoy density | done earlier by the team | ≤ +0.00003 | **Deprioritised.** `work/dshift.txt` and `dshift_v6k.txt`: at 1.4–2× decoys the best cutoff stays 0.75–0.8. |
| 6 | Reproducibility before the final zip. `src/india_addr.py` has PR_MIN=0.6 but loads the v15 models. `stage2.py` needs `feat_cache.py`, `samerule.py`, `dshift_v6k.py` and `v10/cascade_lists.py`, which live in `work/`; move them into `src/`. | whoever packages | required | Todo. **Wait for the peer's v16 code**: it changes `src/india_addr.py` and `src/stage2.py`, and editing them here now would conflict. |

## Tested and dropped (do not repeat)

- **Empty-address rescue v2 (resc2 pairs + LightGBM, with or without tie features):** negative once false merges are counted properly. See `work/release_v18_rescue2_REJECTED.md`, which also covers the evaluation trap to avoid.

- **Accepting stage-2 rejections by pattern (addressed records, US/India):** 11,639 rejected top-1 pairs, 19.6% true. With patterns selected on one half of the validation S1s, none with ≥ 30 pairs reached 85% precision on the other half (`work/v18/rej_cells.py`).
- **Filling in the missing French region from the city.** 32% of France Source-2/3 records have no region, but acceptance is the same with and without it (59.6% vs 59.5%).
- **Breaking empty-address name ties** by addressed-match count, whether picking the most (worse than random) or the fewest (51% vs 47% random for two-way ties). A LightGBM resolver lowered F at every cutoff. Legal form is the best single signal (60%), and that is still not enough.
- **Per-Source-1 expected-F0.5 decision rule:** −0.00015 val.
- **France "equal house number, same name, same street" flips:** validation shows stage 2's rejections there are mostly right (flip precision 0–0.2 below p2 = 0.3).
- **From the team notes:** per-country or per-segment thresholds (≤ +0.00006), phone numbers in names, entity ID order, mass-accepting French empty-address records (LB 0.850), bigger models (overfit), relaxed cutoff for siblings, spreading a match to identical names, letter case, US/India swaps of two real words.
