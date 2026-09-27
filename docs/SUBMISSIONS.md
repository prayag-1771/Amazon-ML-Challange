# Submissions

Validation F0.5 is measured on the 10% held-out Source-1 entities. From v9 onwards it is the stage-2 5-fold CV on those entities. Validation covers only US and India, because France has no labels.

The public leaderboard (LB) scores below are the ones recorded in the release notes, in the team notes pasted on 2026-09-27, and in `work/_status.txt`. **Blank means the score was not recorded in this repo.** The original `SUBMISSIONS.md` was not copied from the old machine; please fill in the gaps.

Output hashes are in `submissions/vN/MANIFEST.txt`. The local copies in `work/sub*/` matched those hashes byte for byte on 2026-09-27.

| Version | Date | Val F0.5 | Public LB | What changed | Local outputs |
|---|---|---|---|---|---|
| v1 | 09-24/25 | 0.527 | | Exact-key baseline | `work/sub1_baseline/` |
| v2 | 09-25 | 0.9771 | | TF-IDF + e5 blocking, prune, LightGBM `lgb_v1` | `work/sub2_lgb_v1/` |
| v3 | 09-25 | 0.9786 | | `lgb_v2` trained on all training queries | `work/sub3_lgb_v2/` |
| v4 | 09-25 | 0.9885 | 0.9822 | Cross-encoder (e5-small) logit as a feature | `work/sub4_lgb_v3/` |
| v5 | 09-25 | 0.9890 | 0.983638 | Label-free distractor-token statistics | `work/sub5_lgb_v4a/` |
| v6 | 09-26 | 0.9893 | 0.984138 | Cross-fitted cross-encoders, `lgb_v5cf` | `work/sub6_lgb_v5cf/` |
| v7 | 09-26 | 0.9893 | 0.9844 | Context-keyed token statistics, `lgb_v6k` | `work/sub7_lgb_v6k/` |
| v8 | 09-26 | 0.9893 | | Ensemble of v5cf + v6k, per-country thresholds | `work/sub8_ens_calib/` |
| v9 | 09-26 | 0.9895 | 0.984799 | Stage-2 collective re-ranker | `work/sub9_stage2/` |
| v10 | 09-26 | 0.9898 | | Structure and sibling features (sibling features hurt on test) | `work/sub10_stage2/` |
| v11 | 09-27 | 0.9899 | | Hard-pair cross-encoder, shift-symmetry cap | `work/sub11_stage2/` |
| probe11 `fr_vfix` | 09-27 | – | **0.9879** | v11 + France variant-vocabulary rule (France-only change) | `work/probe11_fr_vfix/` |
| v12 | 09-27 | 0.98989 | | Cascade candidate cut (6.9M pairs), 3-seed stage 2, France variant and legal-add rules | `work/sub12_cascade/` |
| v13 | 09-27 | 0.99009 | **0.98824** | Empty-address name rescue | `work/sub13_rescue/` |
| v14 | 09-27 | 0.99019 | | Exact house-number shift features in stage 2 | `work/sub14_ndiff/` |
| v15 | 09-27 | 0.99055 | | India addressed no-candidate channel (cutoff 0.9) | `work/sub15_indiaaddr/` |
| probe `fr_empty` | 09-27 | – | 0.850 | Mass-accepting French empty-address matches (team notes). Shows that this idea fails badly. | `work/probe_fr_empty/` |
| v16 (draft) | 09-27 13:13 | | | v15 + India channel at cutoff 0.6 with the v15 models (+4,136 pairs). Superseded by the peer session's v16 (third retrieval method + retrained models, notes: +0.00014 val), which is not on this machine. | `work/sub16_base/` |
| **v17** | 09-27 | 0.99055 (unchanged) | not uploaded | v15 + 806 France pairs from the number-missing rule. Expected LB about +0.00004. Validator PASS. | `work/sub17_fr_on_v15/` |

## v17 hashes (built, not uploaded)
```
53289b84bc77c38548099b1598a05c8e7d75e8e51ed56ecc09c7851fdaafa6d0  output/matching_results.tsv
a01ffc77819cf3dc77cf047945a87afcb16d438bcc41f4086ae078da8c9afe6f  output/candidate_pairs.tsv   (identical to v15)
```
