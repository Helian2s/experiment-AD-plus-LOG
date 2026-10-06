# AD and LoG experiments

Experiments compare an unchanged YOLO11n detector with no preprocessing,
anisotropic diffusion (AD), and AD followed by LoG modulation on COCO val2017.
The corrected version, `02`, also includes a matched-brightness control.

Each experiment folder represents a code version; executions and their results
live in its `runs/` subfolders. Start with the
[experiment index](experiments/README.md) for status, provenance, and run
instructions, or the [experiment trace](experiments/TRACE.md) for history.

| Code version | Run | Results |
| --- | --- | --- |
| [01](experiments/01/README.md) | Archived [5-image pilot](experiments/01/runs/pilot_5/README.md). | [Tables](experiments/01/runs/pilot_5/RESULTS.md), [JSON](experiments/01/runs/pilot_5/summary.json) |
| [01](experiments/01/README.md) | Archived [5,000-image full run](experiments/01/runs/full_5000/README.md). | [Tables](experiments/01/runs/full_5000/RESULTS.md), [JSON](experiments/01/runs/full_5000/summary.json) |
| [02](experiments/02/README.md) | Completed corrected run on all 5,000 val2017 images. | [Tables](experiments/02/runs/full_20261006T031903Z_33e4b749/RESULTS.md), [JSON](experiments/02/runs/full_20261006T031903Z_33e4b749/summary.json) |

All three recorded research runs expose the same compact `summary.json`
structure and `RESULTS.md` table layout inside their run directories.
Experiment 02 also retains its original, detailed run artifacts. Uniform
formatting does not make different protocols scientifically interchangeable:
the two legacy runs have known class-mapping and image-degradation problems and are not
validated COCO benchmark results. Their recorded values remain unchanged;
corrections are isolated in 02. Differences between legacy and corrected
scores are not evidence of an AD or LoG improvement. Saved run metadata and
source snapshots retain their original identifiers; current folder names are
`01` (formerly `001_legacy`) and `02` (formerly `003_corrected_full`).

COCO data and model weights live under `shared/`, which is entirely ignored
by Git. The shared Python environment is `.venv/` at the repository root;
none of these inputs
are duplicated into experiment folders.
The root [run_experiment.py](run_experiment.py) launches experiment
`02`. Research instructions and the instructional notebook are
local supplemental materials under `docs/`, which is ignored by Git and not
included in this repository. The notebook is not evidence of the archived runs.

Check all inputs without inference, then run a three-image smoke test from
the repository root:

```bash
.venv/bin/python run_experiment.py --check-only
.venv/bin/python run_experiment.py --smoke-test
```

The full run is `.venv/bin/python run_experiment.py` and processes every
annotated val2017 image. By default, each inference run creates a new output
directory inside experiment 02; existing output is never overwritten. See the
[protocol and analysis plan](experiments/02/README.md) before
starting the full run.
