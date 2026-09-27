# Amazon ML Challenge 2026: Business Entity Resolution (team Greedy_Decoders)

**Final version: v17**, in `work/sub17_final/`, with validation F0.5 0.99068. See [docs/SUBMISSIONS.md](docs/SUBMISSIONS.md).

For every Source-1 business record, find all Source-2 and Source-3 records of the same business. The score is macro F0.5 per Source-1 entity, with entities that have no match included. The test set adds France, which has no training labels.

| Where | What |
|---|---|
| [business_entity_resolution/](business_entity_resolution/) | The pipeline (`src/`), how to reproduce it, and pinned requirements. It becomes `code/` in the final zip. |
| [docs/SUBMISSIONS.md](docs/SUBMISSIONS.md) | Every version: validation score, leaderboard score, what changed, output hashes |
| [docs/PLAN.md](docs/PLAN.md) | The current improvement plan and the status of each step |
| [docs/WORKLOG.md](docs/WORKLOG.md) | Dated log of what was done, found and decided |
| [docs/analysis/](docs/analysis/) | Analysis write-ups (loss decomposition, score ceiling, France) |
| [Documentation_template.md](Documentation_template.md) | Methodology write-up for the final zip |
| [tools/package.py](tools/package.py) | Builds `dist/<version>/Greedy_Decoders_submission.zip` and `submissions/<version>/MANIFEST.txt` after running the validator |
| [submissions/](submissions/) | `vN/MANIFEST.txt`: SHA-256 of each uploaded version's output files |
| [work/](work/) | Experiment scripts, logs and release notes. `work/v10/` holds the v10–v16 experiments, `work/v17/` the v17 France rule, `work/analysis_0927/` the 2026-09-27 analysis. |

## What is not in git

The `.gitignore` keeps data and large artifacts out:

- the dataset zip and `student_resource/`;
- `output/`;
- everything under `work/` that is parquet, npy, tsv, lgb or model weights (about 54 GB);
- `.venv/`.

Scripts in `work/` read these caches from the local `work/` folder, so they need a machine that already has the caches.

## Environment

Python 3.11. Install with `pip install -r business_entity_resolution/requirements.txt`, adding the CUDA build of torch for the GPU stages. Run the `work/` scripts from `work/`. Running the pipeline end to end needs about 48 GB RAM and a CUDA GPU; see the pipeline README.
