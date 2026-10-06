# AD and LoG experiments

Experiments compare an unchanged YOLO11n detector with no preprocessing,
anisotropic diffusion (AD), and AD followed by LoG modulation on COCO val2017.
The corrected third experiment also includes a matched-brightness control.

Each experiment has its own code and results in one folder. Start with the
[experiment index](experiments/README.md) for status, provenance, and run
instructions, or the [experiment trace](experiments/TRACE.md) for history.

| Folder | Status |
| --- | --- |
| [001_pilot_5](experiments/001_pilot_5/README.md) | Archived 5-image result; accompanying code reconstructed afterward. |
| [002_full_5000](experiments/002_full_5000/README.md) | Archived 5,000-image result and log; accompanying code reconstructed afterward. |
| [003_corrected_full](experiments/003_corrected_full/README.md) | Corrected protocol for all 5,000 val2017 images; full experiment not yet run. |

The archived scores have known class-mapping and image-degradation problems
and are not validated COCO benchmark results. Their code and artifacts remain
unchanged by this preparation; corrections are isolated in experiment 003. Its future scores must
not be compared with the legacy scores as evidence of a method improvement.

COCO data and model weights live under `shared/`, which is entirely ignored
by Git. The shared Python environment is `.venv/` at the repository root;
none of these inputs
are duplicated into experiment folders.
The root [run_experiment.py](run_experiment.py) launches experiment
`003_corrected_full`. Research instructions and the instructional notebook are
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
directory inside experiment 003; existing output is never overwritten. See the
[protocol and analysis plan](experiments/003_corrected_full/README.md) before
starting the full run.
