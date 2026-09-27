## Submission v17 (final): v16 India channel (cutoff 0.7) + France number-missing rule (val F0.5 0.99068)

**Upload `work/sub17_final/matching_results.tsv`.** Expected LB gain over v15 is about +0.0002: India +0.00015, France +0.00004.

| Build | Contents | Matches | Candidates |
|---|---|---|---|
| `work/sub16_india07/` (v16) | v15 with the India addressed channel re-scored by the retrained `india_addr16` models at cutoff 0.7: 8,918 pairs instead of 5,505 | 5,827,973 | 6,921,864 |
| `work/sub17_final/` (v17) | v16 + the France patch below | **5,828,779** | 6,921,864 |

Both pass the validator. `src/` reproduces v17: `python -m src.stage2 predict` with `src/france_rules.py`.

The sections below describe the France part. They were first built on v15 as `work/sub17_fr_on_v15/` ("v17a").

### Changes vs v15
A France top-1 pair that v15 left unmatched is now accepted when all of these hold:
- the S2/S3 record has **no house number** while its S1 has one;
- the name core is identical (a legal form may be added or dropped, but not changed);
- the street name is the same up to typos (fuzz.ratio ≥ 85 after removing city/region words, street types and stopwords);
- stage-2 p2 ≥ 0.01;
- exactly one France S1 has this name and street name (ties removed).

That adds **806 pairs**: legal form added 438, same 293, dropped 75.

Code: `work/v17/fr_rule.py` (rule), `work/v17/build.py` (applies patches and runs the validator).

### Why
- In US/India validation this pattern is **100% true matches** (15,374 pairs), and stage 2 rejects only 1 of them, which was a true match.
- On test, stage 2 rejects **3.3%** of the same pattern in France (32,853 pairs). That is about 480× the validation rate.
- The equal-number version of the pattern is **not** flipped. Its France rejection rate is only 2.3× validation, and validation shows those rejections are mostly right.

### Validation
France has no labels, so validation is unchanged: 0.99055 (US 0.99087, India 0.99007).

### Test-side
| | Matches | Candidates |
|---|---|---|
| v15 | 5,824,560 | 6,918,451 |
| v17 | **5,825,366** | 6,918,451 (identical file) |

The 806 added pairs touch 754 France S1s, 24 of which had no match before.

Expected LB delta by assumed precision of the added pairs:

| Precision | LB delta |
|---|---|
| 1.0 | +0.000047 |
| 0.9 | +0.000033 |
| 0.8 | +0.000019 |

Validator: **PASS** with `--check-ids`.

### SHA-256 (final, work/sub17_final)
```
456a67cb7a3454f9752bbaa76aaf23ef84665573913140aa85475c192a807ea3  output/matching_results.tsv
35013d3a2bfce19cd06b6acd04c7a8a6cdaba647aaf3958320a1d839d0d88869  output/candidate_pairs.tsv
```

### SHA-256 (v17a, work/sub17_fr_on_v15)
```
53289b84bc77c38548099b1598a05c8e7d75e8e51ed56ecc09c7851fdaafa6d0  output/matching_results.tsv
a01ffc77819cf3dc77cf047945a87afcb16d438bcc41f4086ae078da8c9afe6f  output/candidate_pairs.tsv
```

### Apply on top of v16
The patch touches France records only, so it can go on top of any base:
```
cd work && python v17/build.py <v16_dir> <out_dir> v17/fr_patch.parquet
```
