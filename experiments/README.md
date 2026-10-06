# AD and LoG experiment index

Each direct child folder represents one code version and owns its source and
README. Executions of that version live under `runs/`, each with its own
configuration, outputs, and provenance. Changing image count or starting a
pilot/full execution creates a new run, not a duplicate code-version folder.
Dataset files, weights, and the virtual environment are shared rather than copied.

## Experiments

| Code version | Runs | Code and result status |
| --- | --- | --- |
| [01](01/README.md) | [pilot_5](01/runs/pilot_5/README.md), [full_5000](01/runs/full_5000/README.md) | Shared reconstructed legacy code; archived 5-image and 5,000-image results, plus the full-run log. |
| [02](02/README.md) | [Completed full run](02/runs/full_20261006T031903Z_33e4b749/RESULTS.md); local smoke runs | Corrected code; 5,000-image full result with original detailed metrics and execution-time source snapshot. |

The former `001_pilot_5` and `002_full_5000` folders are now the two runs of
`01`. Their reconstructed scripts differed only in image count and
its docstring. The current code-version folders are named `01` and `02`.
Saved run metadata and source snapshots retain their historical identifiers
(`001_legacy` and `003_corrected_full`); new runs record the current names.

The historical JSON and log files were moved without changing their bytes.
The log's local project-root prefix was subsequently replaced with
`<PROJECT_ROOT>` for publication; all other log bytes and both JSON files
remain unchanged. The [trace](TRACE.md#public-repository-cleanup-on-2026-10-05)
records this privacy edit and the current artifact fingerprints.
The original launch commands and source snapshots were not preserved.
The scripts originally beside the two historical results were reconstructed during the
2026-10-05 reorganization from the then-current source and recorded image
counts. Their processing and evaluation algorithms retain the known defects;
paths and overwrite protection were adapted to this layout. Their consolidated
shared script selects image count through `--images`. It does not prove
the exact code or environment that produced the archived scores.

The source used for reorganization, before those adaptations, had SHA-256
`bed8d4ed0b715c2703b935a50bcdc0f700fda139974826ba35f3861527cec435`.
This identifies the reconstruction basis, not either historical execution.

## Common result format

Each recorded research run has a derived `summary.json` and generated
`RESULTS.md` inside its run directory. All three JSON summaries use
`map_results` (per-condition, per-method
`mAP_50_95` and `mAP_50`), `corr_summary` (per-condition `mean` and `std`),
and `params` (the seven common AD/LoG, image-count, and seed parameters).
Each run's original `results.json` remains byte-for-byte unchanged. The compact
02 summary is derived from its completed run; detailed metrics, validity counts,
full configuration, hashes, and provenance remain in that run's directory.
The compact schema is not a complete reproduction manifest. Legacy runs have
retrospective `run_config.json` files derived from archived parameters, not
recovered execution manifests.

The Markdown reports have identical headings and table columns. The brightness
control appears only in experiment 02's JSON; older reports mark it `Not run`,
not zero. Formatting neither invents missing experiments nor changes recorded
values. A shared format does not remove the legacy protocol defects.

To regenerate the derived summaries and reports without detector inference:

```bash
.venv/bin/python format_results.py --run-dir experiments/02/runs/full_20261006T031903Z_33e4b749
```

Add `--check` to verify that generated files match their sources without
writing them. The formatter leaves the two historical JSON files and original
02 run artifacts untouched.

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
shell's working directory. Both code versions create an exclusive directory
under their own `runs/` for each new execution and refuse to reuse an existing
directory. Experiment 02 records code snapshots, metadata, diagnostics, and results.
The root [run_experiment.py](../run_experiment.py) launches
`02`; it does not select a historical run automatically.

## Running and adding experiments

From the repository root, use the existing environment:

```bash
.venv/bin/python run_experiment.py --check-only
.venv/bin/python run_experiment.py --smoke-test
.venv/bin/python run_experiment.py
```

These commands respectively validate inputs without inference, run a separate
three-image technical smoke test, and run all 5,000 annotated val2017 images.
Do not interpret smoke-test metrics as research results. Experiment 02
compares four methods under six conditions with corrected evaluation and
explicit detector settings; read its [protocol](02/README.md).
A full corrected run has completed; running the last command again starts a
new run rather than displaying its results. The two archived legacy runs
must not be overwritten. For another execution of the reconstructed legacy
version, see [its run options](01/README.md#additional-runs).

Experiment 02 records raw detections locally for re-evaluation and later
paired bootstrap analysis. They and default smoke-run directories are ignored
by Git; aggregate full-run artifacts remain eligible for publication.

The root `.venv/` is shared by experiments with compatible dependencies.
It is excluded from Git. If a future experiment needs incompatible package
versions, use a separate environment and record its dependencies; changing
the shared environment does not reproduce the historical runs automatically.

For another run of the same code, use that version's run options; keep the
outputs in a new subfolder of its `runs/`. Do not copy the version folder just
to change the image count or to progress from pilot to full evaluation.

For an algorithm/protocol change, copy the complete source folder to a new
direct child of `experiments/`, excluding `runs/` and caches. Experiment 02 uses multiple
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

The compact summaries share the schema described above. Scores are fractions,
not percentages. For experiment 02, use the original run artifacts when
analyzing metrics beyond that common subset.

The known `cls + 1` mapping does not match sparse COCO category IDs, and
`low_light` with gamma 0.5 brightens images. These defects affect evaluation
and the interpretation of `low_light` and `multifactor`. Archived scores are
preliminary evidence, not validated quality measurements or confirmation of
the research hypothesis. The shared legacy script retains these behaviors;
experiment 02 corrects them in a separate protocol. Changes in evaluation,
degradation, and inference settings prevent treating the legacy-to-corrected score
difference as the benefit of AD or LoG.

See the [trace](TRACE.md) for the recorded processing sequence, historical
metadata, missing provenance, and artifact fingerprints.
