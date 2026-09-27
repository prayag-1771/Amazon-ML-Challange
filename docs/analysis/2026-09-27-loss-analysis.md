# Loss analysis, 2026-09-27 (against v15)

Scripts are in `work/analysis_0927/` and run from `work/`. The numbers are validation F0.5 on the 10% held-out Source-1 entities unless marked "LB" or "test".

## 1. Missing values by country and source (`missprof.py`)

| | Test France S2/S3 | Test India S2/S3 | Test US S2/S3 | Train (US/India) S2/S3 |
|---|---|---|---|---|
| Empty address | 3.0% | 2.3–2.5% | 2.8–2.9% | 2.9–3.7% |
| No state/region detected | **31–33%** | 0% | 0.1% | 0–0.1% |
| No house number | 3.7% | 4.7–5.7% | 4.2–5.0% | 5.2–6.9% |
| "null" placeholder in address | 0% | 2.8–2.9% | 3.7–3.9% | 2.8–3.9% |

- **France region.** France Source-2/3 addresses are often just "street, city". The region is really missing, not badly parsed.
- **US addresses.** US Source-1 addresses have about 3 parts, versus about 6 in India, and contain no ZIP codes.
- **Test composition.** Test Source-1 is France 15.0%, India 46.8%, US 38.3%. Training is US 60%, India 40%.

## 2. Does missingness hurt? (`cells.py`)

| Cell (Source-2/3 record) | Train true-match rate | v15 val recall | Test acceptance US / India / France |
|---|---|---|---|
| Empty address | **97.7–97.8%** | 0.53–0.54 | 60% / 52% / **47%** |
| No house number | 97% | 0.99 | 97% / 96% / **86%** |
| No state | – | – | France 59.6% vs complete records 59.5% |
| Complete | 72% | 0.99 | 57% / 56% / 59% |

- **Only true matches lose address parts.** The generator removes address parts only from genuine variant records, never from decoys.
- **France's missing region is harmless.** Its acceptance is the same as for complete records.
- **France records with no house number are under-accepted.** See section 5.

## 3. Where validation loss comes from (`lossdec.py`)

v15 validation F0.5 is 0.99055, so the loss is 0.00945. Each gain below assumes that category is fixed completely.

| Category | Pairs | Gain |
|---|---|---|
| Empty address, name shared by ≥ 2 Source-1 businesses | 13,912 | 0.00566 |
| Has an address, missed by blocking/cascade (86% of these have no candidate at all) | 3,065 | 0.00140 |
| Has an address, rejected below cutoff | 2,408 | 0.00104 |
| Empty address, unique name | 1,783 | 0.00067 |
| False merges (record belongs to another business, or is a decoy) | 776 | 0.00089 |

**Validation ceiling:** fixing every category except the shared-name ties gives 0.99449.

Split of the empty-address misses (`verify_notes.py`):

| Group | Pairs | Gain |
|---|---|---|
| A: no cascade candidate | 4,949 | +0.00205 |
| B: found but below cutoff | 7,246 | +0.00290 (US: 5,228 pairs, +0.00357 on the US score) |
| C: only wrong candidates | 3,500 | +0.00144 |

- **Group B is mostly ties:** 92% of the US pairs have a name shared by two or more Source-1 businesses.
- **Rescue acceptance in group A** is 3.3% (US) and 15.5% (India).
- **Only a small part of A is winnable:** truly unique names are about 430 pairs per country (+0.0003).

## 4. Why the ties can't be broken (`empty.py`, `ambig.py`, `verify_notes.py`)

Accuracy of picking the true owner among Source-1 businesses with the same `name_core`, for a two-way tie:

| Rule | Accuracy |
|---|---|
| Random | 0.472 |
| Most addressed matches | 0.441 |
| Fewest addressed matches | 0.506 |
| Legal-form match | 0.603 |
| Legal form, then fewest matches | 0.627 |

- **As an owner prior**, P(business owns an empty-address record) is 0.23 with 1 addressed match and 0.08 with 8.
- **The LightGBM resolver lowered F at every cutoff**, because v15's rescue already takes the resolvable cases.
- **Estimated test-side cost** (`ceiling.py`): France 0.0049, India 0.0033, US 0.0027. Weighted by test share that is **0.0033 on the LB, so the ceiling is about 0.9967**.

## 5. France (`france.py`, `nonum.py`, `work/v17/`)

- **France F0.5 is about 0.979**, worked back from the v13 LB and assuming US/India score on test as they do on validation. US/India are about 0.990.
- **Stage 2 is much less sure on France.** 9.8% of France records have a top-1 p2 between 0.05 and 0.95, versus 2.0% (India) and 2.9% (US).
- **The first cell audit overstated the problem.** Its street test counted "rue" as a shared word, and that made France look 30–300× off.
- **Result after fixing the street test and excluding Compagnie/Cie swaps:**
  - *Number missing on the query, same name and street:* validation purity 1.000 (15,374 pairs), France rejects 3.3%, which is 480× more. This became the **v17 patch of 806 pairs**.
  - *Equal number, same name and street:* France rejects 0.17% vs 0.07%. Validation flip precision is only 0.09, so these are **not flipped**.

## 6. v16 India cutoff (`work/prsweep16.py`, retrained channel models)

| Cutoff | Val F0.5 | Precision on added pairs |
|---|---|---|
| 0.5 | 0.99070 | 0.904 |
| 0.6 | 0.99070 | 0.931 |
| 0.7 | 0.99068 | 0.949 |
| 0.8 | 0.99066 | 0.973 |
| 0.9 | 0.99057 | 0.994 |

The channel retrieves the true business for only 61.8% of the India no-candidate misses. Of those, 714 are ranked first but score below 0.6.

## 7. Test set has more decoys

The test set has about 5.8 Source-2/3 records per Source-1 business, versus 4.7 in training, while accepted pairs per business are similar (3.3–3.4). About 1.1 extra unmatched records per business means wrong merges weigh more on the LB than validation suggests.
