# AD and LoG experiment index

Each direct child folder owns its code and recorded results when available.
The three named experiments use `run_experiment.py` and a README describing
status and provenance. Dataset files, weights, and the virtual environment
are shared rather than copied.

## Experiments

| Folder | Images | Code and result status |
| --- | ---: | --- |
| [001_pilot_5](001_pilot_5/README.md) | 5 | Archived [result](001_pilot_5/results.json); reconstructed code, not an original execution snapshot. |
| [002_full_5000](002_full_5000/README.md) | 5,000 | Archived [result](002_full_5000/results.json) and [log](002_full_5000/run.log); reconstructed code. |
| [003_corrected_full](003_corrected_full/README.md) | 5,000 | Corrected protocol prepared; no completed full result yet. |

The historical JSON and log files were moved without changing their bytes.
The log's local project-root prefix was subsequently replaced with
`<PROJECT_ROOT>` for publication; all other log bytes and both JSON files
remain unchanged. The [trace](TRACE.md#public-repository-cleanup-on-2026-10-05)
records this privacy edit and the current artifact fingerprints.
The original launch commands and source snapshots were not preserved.
The scripts beside the two historical results were reconstructed during the
2026-10-05 reorganization from the then-current source and recorded image
counts. Their processing and evaluation algorithms retain the known defects;
paths and overwrite protection were adapted to this layout. They do not prove
the exact code or environment that produced the archived scores.

The source used for reorganization, before those adaptations, had SHA-256
`bed8d4ed0b715c2703b935a50bcdc0f700fda139974826ba35f3861527cec435`.
This identifies the reconstruction basis, not either historical execution.

## Shared inputs

Paths below are relative to the repository root:

```text
shared/coco/images/val2017/
shared/coco/annotations/instances_val2017.json
shared/yolo11n.pt
.venv/
```

The entire `shared/` directory is excluded from Git. It contains local inputs,
not experiment code or results; those stay in the folders listed above.

Scripts resolve shared data and model paths from their own locations, not the
shell's working directory. The two archived scripts protect their existing
`results.json`. Experiment 003 instead creates an exclusive directory, by
default under its own `runs/`, with code snapshots, metadata, diagnostics, and results.
The root [run_experiment.py](../run_experiment.py) launches
`003_corrected_full`; it does not select a historical run automatically.

## Running and adding experiments

From the repository root, use the existing environment:

```bash
.venv/bin/python run_experiment.py --check-only
.venv/bin/python run_experiment.py --smoke-test
.venv/bin/python run_experiment.py
```

These commands respectively validate inputs without inference, run a separate
three-image technical smoke test, and run all 5,000 annotated val2017 images.
Do not interpret smoke-test metrics as research results. Experiment 003
compares four methods under six conditions with corrected evaluation and
explicit detector settings; read its [protocol](003_corrected_full/README.md).
The full run has not yet been performed. The two archived folders retain their
legacy implementation and must not be overwritten.

Experiment 003 records raw detections locally for re-evaluation and later
paired bootstrap analysis. They and default smoke-run directories are ignored
by Git; aggregate full-run artifacts remain eligible for publication.

The root `.venv/` is shared by experiments with compatible dependencies.
It is excluded from Git. If a future experiment needs incompatible package
versions, use a separate environment and record its dependencies; changing
the shared environment does not reproduce the historical runs automatically.

For a new experiment, copy the complete source folder to a new direct child
of `experiments/`, excluding `runs/` and caches. Experiment 003 uses multiple
Python modules, so copying only `run_experiment.py` is insufficient. Update the
new README and configuration before running, recording the hypothesis,
changes, image count, and comparisons. Keep code and results together; after
collecting results, make algorithm changes in another experiment folder.
Do not copy old results into a new experiment.

The Ubuntu notebook (`docs/AD_LoG_Experiment_Ubuntu.ipynb`, relative to the
repository root) is local supplemental material excluded from this repository.
It is a separate instructional implementation, not an execution record for
either archive.
Its future saves create their own folders under `experiments/`, containing
results, metadata, and `notebook_source.py` exported from the saved notebook
code cells. Save the notebook before running; the export is not a guarantee
that saved cells exactly match the code executed in a live kernel.

## Result interpretation

Both archived JSON files contain `map_results` (mAP at IoU 0.50:0.95 and 0.50),
`corr_summary` (mean and standard deviation of LoG/gradient correlation), and
`params`. Scores are fractions, not percentages.

The known `cls + 1` mapping does not match sparse COCO category IDs, and
`low_light` with gamma 0.5 brightens images. These defects affect evaluation
and the interpretation of `low_light` and `multifactor`. Archived scores are
preliminary evidence, not validated quality measurements or confirmation of
the research hypothesis. The archived scripts retain these behaviors;
experiment 003 corrects them in a separate protocol. Changes in evaluation,
degradation, and inference settings prevent treating the 002-to-003 score
difference as the benefit of AD or LoG.

See the [trace](TRACE.md) for the recorded processing sequence, historical
metadata, missing provenance, and artifact fingerprints.
