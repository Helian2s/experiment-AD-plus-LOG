# Experiment 01: reconstructed legacy AD and LoG protocol

This folder owns one code version, with multiple executions under `runs/`.
The former `001_pilot_5` and `002_full_5000` scripts differed only in their
image-count setting and docstring. They have been consolidated into the shared
[run_experiment.py](run_experiment.py); no processing or evaluation algorithm
was changed. Experiment 02 remains a separate corrected code version.

## Archived runs

| Run | Images | Results | Provenance |
| --- | ---: | --- | --- |
| [pilot_5](runs/pilot_5/README.md) | 5 | [Tables](runs/pilot_5/RESULTS.md), [summary](runs/pilot_5/summary.json), [original JSON](runs/pilot_5/results.json) | No execution log or original source snapshot. |
| [full_5000](runs/full_5000/README.md) | 5,000 | [Tables](runs/full_5000/RESULTS.md), [summary](runs/full_5000/summary.json), [original JSON](runs/full_5000/results.json), [log](runs/full_5000/run.log) | Historical log, with its previously documented privacy redaction. |

The archived JSON files and sanitized log retain their exact bytes. Each run's
`run_config.json` is retrospective metadata copied from its original result
parameters, not a recovered historical launch manifest. Its `summary.json` and
`RESULTS.md` are derived presentation files in the common result format.
The folder was renamed from `001_legacy` to `01`; archived configuration
records retain the earlier identifier, while new executions record `01`.

## Code provenance and limitations

The shared script was reconstructed on 2026-10-05 from the then-current
implementation. Neither historical run's exact execution-time code, launch
command, or full environment was preserved. Consolidating matching
reconstructions does not establish exact historical reproduction.

The known legacy defects remain, including incorrect `cls + 1` class mapping,
gamma 0.5 brightening under the `low_light` label, periodic AD boundaries,
implicit detector settings, and evaluation-error handling. These results are
not corrected COCO benchmark measurements or proof of the research hypothesis.
Raw detections were not preserved for the historical runs.

## Additional runs

From the repository root, these commands execute the same reconstructed legacy
code with different image counts. They do not display or reproduce the archived
results automatically, and the legacy protocol is not recommended for new
scientific conclusions:

```bash
.venv/bin/python experiments/01/run_experiment.py --images 5
.venv/bin/python experiments/01/run_experiment.py --images 5000
```

Each execution creates a unique directory under this version's `runs/`, saves
the selected parameters and script snapshot there, and refuses to reuse an
existing directory. `--run-name NAME` selects an explicit new directory name;
`pilot_5` and `full_5000` are reserved by the archives and cannot be overwritten.
The default count is 5,000 and the seed remains 42. Shared inputs and the Python
environment remain at the repository root.

Change algorithms in a new code-version folder; change execution settings by
starting another run within the same version. See the [index](../README.md)
and [trace](../TRACE.md) for the corrected protocol and archival history.
