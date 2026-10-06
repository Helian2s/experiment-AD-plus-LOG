# AD and LoG experiment trace

Recorded on 2026-10-05 from the repository history, saved result files, full-run
log, current source, and inspected notebook cells. This is a retrospective
record, not an original shell transcript. It covers the experiment artifacts
present here and the subsequent work to archive them.

Two runs are preserved: a 5-image preliminary result and a completed
5,000-image run. The full run used the original class-mapping and gamma
behavior, so the recorded scores are not a corrected COCO benchmark.

## Project reorganization on 2026-10-05

After the original trace and results archive were created, the project was
reorganized so each experiment owns its code and output in one folder:

| Previous location | Location after that reorganization |
| --- | --- |
| `results/results.json` | [001_pilot_5/results.json](001_pilot_5/results.json) |
| `results/full_5000.AfmLxA/results.json` | [002_full_5000/results.json](002_full_5000/results.json) |
| `results/full_5000.AfmLxA/run.log` | [002_full_5000/run.log](002_full_5000/run.log) |
| `experiment/run_experiment.py` | `003_current_500/run_experiment.py`, subsequently replaced by [003_corrected_full](003_corrected_full/README.md). |
| `results/README.md` | [README.md](README.md) |
| `results/TRACE.md` | This file. |

The archived JSON and log files were moved byte-for-byte. The log's local
project-root prefix was later redacted as described below. Historical relative
paths inside the log and the archival narrative are retained; they describe
the layout at that time, not current run instructions.

The two historical folders now contain reconstructed scripts based on the
source available at reorganization, with `N_IMAGES` set to the recorded 5 or
5,000. No exact execution-time source was recovered. The scripts preserve
the processing and evaluation algorithms while updating paths and preventing
an existing `results.json` from being overwritten. They must not be treated
as original source snapshots or evidence of exact historical reproduction.

The then-current 500-image configuration received its own folder but had no saved result.
The old script path initially became a compatibility launcher for that configuration.
At this stage, shared data, weights, and the environment remained under `experiment/`.
The reorganization did not rerun experiments or correct their methodology.

## Shared environment relocation on 2026-10-05

The existing Python 3.12.3 environment was subsequently relocated from
`experiment/.venv/` to the repository-root `.venv/`. Activation scripts and
virtual-environment configuration were regenerated without installing pip or
changing package versions; installed command launchers were updated for the
new absolute path. PyCharm now uses the root environment. Data, model weights,
experiment code, and archived outputs were not moved by this follow-up change.
Historical environment-path mentions below describe the earlier layout.

## Root launcher relocation on 2026-10-05

The launcher was subsequently moved from `experiment/run_experiment.py` to
the repository-root [run_experiment.py](../run_experiment.py). Its target at that time was
`experiments/003_current_500/run_experiment.py`; each experiment's implementation
and results remain together in its own folder. The old launcher path was
removed. Documentation and the launcher test were updated; no inference ran.

## Shared inputs rename on 2026-10-05

After moving the launcher and Python environment to the root, the remaining
`experiment/` directory was renamed to `shared/`. It contains the COCO dataset
and YOLO11n weights and is now entirely excluded from Git. Runtime paths,
documentation, notebook paths, and layout tests were updated. Archived
outputs and algorithms were not changed; old paths below remain historical.

## Public repository cleanup on 2026-10-05

For publication, the local absolute project-root prefix in `run.log` was
replaced with `<PROJECT_ROOT>` in exactly two places, including `RUN_INFO`.
The historical result-directory suffix, progress output, line endings,
timings, metrics, and every other log byte were preserved. Both result JSON
files remain byte-for-byte original. The log fingerprint below identifies
the sanitized public artifact, not the original private log.

Local Git history was rewritten to remove the same path from the historical
log and replace personal author/committer emails with the existing GitHub
`noreply` identity. Commit messages, timestamps, parent topology, and all
other historical file contents were preserved. Commit identifiers below
refer to this rewritten history. A private backup of the original Git data
and log was retained outside the repository; no push was performed.

The `.idea/` directory was added to `.gitignore` and removed from the current
Git index without deleting local IDE settings. Links to the unpublished
notebook were replaced with local-material notes, and its compilation test
now skips when the notebook is absent. No experiment was rerun and no
processing or evaluation algorithm was changed by this cleanup.

## Evidence

- [Current experiment source](003_corrected_full/run_experiment.py): corrected
  experiment 003; not evidence of the code used for either archived run.
- [Reconstructed historical source](002_full_5000/run_experiment.py): legacy
  processing described below; not an original execution-time snapshot.
- [Preliminary results](001_pilot_5/results.json): metrics and parameters for 5 images.
- [Full results](002_full_5000/results.json): metrics and parameters for 5,000 images.
- [Full run log](002_full_5000/run.log): start metadata, progress, evaluation
  output, summary tables, and completion marker.
- [Experiment index](README.md): updated from the archive description added
  with the results commit.

The inspected Ubuntu notebook (`docs/AD_LoG_Experiment_Ubuntu.ipynb`, relative
to the repository root) is local supplemental material, ignored by Git and
not included in this repository. It contains instructions and an adapted
implementation, not execution evidence for the archived results.

## Timeline

Times below use UTC. File modification times are labeled explicitly: they
do not establish when a command was executed or who executed it.

| Time | Recorded event | Evidence |
| --- | --- | --- |
| 2026-09-18 03:07:06 | Repository initialized with `.gitignore`. | Commit `f92ad89`. |
| 2026-09-18 15:17:10 | Experiment script and IDE project files committed. | Commit `8797699`, `initial code commit`. |
| 2026-10-02 23:14:19 | Preliminary JSON last modified; it records `N_IMAGES = 5`. | File modification time and `results.json`; no preliminary run log is saved here. |
| 2026-10-02 23:17:48.941074 | Full run started with 5,000 images. | `RUN_INFO.started_utc` and `RUN_INFO.images`. |
| Approximately 2026-10-03 01:17:18 | Full run completed after 7,169.4 seconds, about 1 hour 59 minutes 29 seconds. | `RUN_FINISHED`; finish time calculated from the recorded start and elapsed duration. |
| 2026-10-03 01:36:42 | Ubuntu notebook modification time observed when this trace was created. | File modification time; this is not evidence of notebook execution. |
| 2026-10-05 23:23:04 | Both JSON files, the full log, README, and a log ignore-rule exception committed locally. | Commit `ee2e636`, `Archive experiment results and document known limitations`. |

## Local preparation before reorganization

The working tree contains COCO images under `experiment/coco/images/val2017`,
annotations under `experiment/coco/annotations`, and weights at
`experiment/yolo11n.pt`. The experiment uses these extracted dataset files
directly. The commands used to prepare the dataset were not preserved in the
reviewed run artifacts.

Before reorganization, the source differed from commit `8797699` only in
these dataset paths:

```text
coco/images/val2017
  -> experiment/coco/images/val2017
coco/annotations/instances_val2017.json
  -> experiment/coco/annotations/instances_val2017.json
```

The IDE module also locally excludes `experiment/.venv` from indexing.
These changes existed before the results were archived and were not included
in the archival commit.
The IDE setting alone does not establish the interpreter used for the full run.

## Recorded run configuration

| Setting | Preliminary run | Full run |
| --- | --- | --- |
| Image count | 5 | 5,000 |
| Random seed | 42 | 42 |
| AD kappa | 30 | 30 |
| AD iterations | 5 | 5 |
| AD time step | 0.15 | 0.15 |
| LoG sigma | 1.5 | 1.5 |
| LoG modulation beta | 0.3 | 0.3 |

Both JSON files contain all six conditions and all three methods. The full
log records 5,000 selected images out of 5,000 available, PyTorch
`2.14.1+cu130`, and `cuda_available: false`. It identifies fused YOLO11n with
100 layers, 2,616,248 parameters, and 0 gradients. The experiment performs
inference with fixed weights; there is no training step in the source.

The source inspected when this trace was first written defaulted to 500 images.
The archived 5-image and 5,000-image runs used different counts from that default;
the exact mechanism used to set them was not saved.

## Processing sequence

The following historical sequence is reconstructed from the source inspected
when this trace was first written and is
consistent with the conditions, method names, and evaluation output in the
full log. An exact source snapshot from the time of execution is unavailable.

1. Seed NumPy with 42, sort the image filenames, and sample the requested
   number without replacement. Use the same selection across conditions.
2. Load and fuse YOLO11n, then load the COCO instance annotations.
3. Process the conditions in the order listed below. For each image, create
   the degraded input once and pass it through three branches: baseline,
   AD only, and AD followed by LoG modulation.
4. Run the detector for each branch and accumulate boxes, scores, image IDs,
   and category IDs in memory.
5. For each image, correlate the normalized absolute LoG map with the squared
   gradient magnitude of the AD output using Pearson correlation.
6. Evaluate the accumulated detections with COCOeval for each of the
   6 conditions and 3 methods. Collect mAP at IoU 0.50:0.95 and 0.50.
7. Calculate the mean and standard deviation of the per-image correlations;
   print the summary tables and save aggregate metrics and parameters to JSON.

| Condition | Transformation in the source | Full-log progress at completion |
| --- | --- | --- |
| `clean` | Copy the input image. | 5,000/5,000; 19:26 |
| `gaussian_noise` | Add Gaussian noise with sigma 25; clip to uint8. | 5,000/5,000; 20:06 |
| `blur` | Apply a 5 by 5 Gaussian blur with sigma 0 passed to OpenCV. | 5,000/5,000; 18:50 |
| `low_light` | Apply `(pixel / 255) ** 0.5` through a lookup table. | 5,000/5,000; 18:43 |
| `haze` | Blend with atmospheric light 0.9 using beta 1.0 and synthetic depth varying from 0.1 to 1.0 across the image width. | 5,000/5,000; 20:19 |
| `multifactor` | Apply noise, then gamma correction, then haze. | 5,000/5,000; 20:42 |

The AD branch applies channel-wise Perona-Malik diffusion. The LoG branch
computes the absolute Gaussian Laplacian of grayscale AD output, normalizes
the map, and modulates brightness as `AD * (1 + 0.3 * LoG_map)`, clipping
the result to uint8. It does not add a fourth channel to the detector.

At 5,000 readable images, this loop entails 90,000 detector calls
(`5000 * 6 * 3`). That is a calculation from the loop structure, not a
separately recorded inference counter; the source can skip unreadable images.

## Saved full-run scores

These are the archived mAP at IoU 0.50:0.95 values, rounded to six decimal
places. They are fractions, not percentages. Full precision, mAP at IoU 0.50,
and correlation summaries are in the linked JSON.

| Condition | Baseline | AD only | AD plus LoG |
| --- | ---: | ---: | ---: |
| clean | 0.054642 | 0.052697 | 0.052340 |
| gaussian_noise | 0.035968 | 0.040257 | 0.040256 |
| blur | 0.051988 | 0.049382 | 0.048653 |
| low_light | 0.052806 | 0.051334 | 0.050811 |
| haze | 0.048344 | 0.045544 | 0.045086 |
| multifactor | 0.029195 | 0.033943 | 0.035100 |

## Known issues and missing provenance

The full log explicitly records two unresolved issues: `cls + 1` does not
correctly map YOLO classes to sparse COCO category IDs, and gamma 0.5
brightens images despite the `low_light` label. The latter also affects
`multifactor`. These outputs preserve that behavior; the archival work did
not fix it or rerun the experiment.

That historical source also uses periodic AD boundaries via `np.roll` and leaves detector
inference options at library defaults. Only one seed is represented in these
artifacts. There are no saved confidence intervals or repeated-seed results.

The reviewed run artifacts do not preserve the launch command, exact source
snapshot, complete dependency versions, Python executable, selected image
order, model checksum, raw detections, or per-image correlations. The full
run's input count is logged, but a skipped-image count is not. These gaps
limit exact reproduction and prevent reevaluating the stored aggregate
scores with a corrected class mapping alone.

The local, unpublished Ubuntu notebook contains a separate workflow with a 5-image
default, unique result directories, additional metadata saving, and stricter
evaluation error handling. Its inspected setup and save cells have no saved
outputs. Its instructions and code do not establish that those steps ran;
in particular, they do not supply missing metadata for the archived runs.

## Archival work performed in this conversation

1. Inspected Git status and identified the existing source and IDE changes,
   dataset files, documentation, weights, and untracked results.
2. Read both result JSON files and the full run log, distinguishing the
   5-image preliminary output from the 5,000-image run.
3. Added `results/README.md` describing the artifacts, recorded parameters,
   known issues, and reproduction limits.
4. Added `!/results/full_5000.AfmLxA/run.log` to `.gitignore` so the original
   log could be tracked despite the general `*.log` rule.
5. Parsed both JSON files and checked expected conditions, methods,
   parameters, finite metrics, metric ranges, and correlation ranges.
   Checked that all six log progress entries reached 5,000/5,000, that the
   completion marker exists, and that no traceback or source-format
   evaluation error appeared. Compared the full-run mAP and correlation
   summary rows against the JSON to the log's four-decimal precision.
6. Ran `git diff --check` and `git diff --cached --check`; both passed.
   Staged only the two JSON files, full log, README, and `.gitignore` change.
7. Created the local archival commit, now identified as
   `ee2e6363dc365887d9b48b9bcdea0c6ccc908776` after the privacy rewrite.
   At the original archival step, the existing JSON files and log were
   preserved without edits. The later log redaction is documented above.
   No push was performed.
8. Created this trace in response to the subsequent request. The experiment
   was not rerun while archiving the results or writing this trace.

Dataset files, model weights, the untracked `docs/` contents, and
the existing source and IDE changes were left out of the archival commit.
This trace was added afterward and is not part of `ee2e636`.

## Artifact fingerprints

SHA-256 values identify the published artifacts: the two original JSON files
and the privacy-sanitized log. These are not checksums recorded at experiment
launch. Paths below use their post-reorganization locations; the move itself
did not change file contents. The log checksum was updated after redaction.

```text
8265a90458cca09c989245e2ec0610aa1f4e674e8e0bef37eb900cf0759fbe8e  experiments/001_pilot_5/results.json
287c188061661b1d7702895b434fc18aa095496fb8ea5c8382a5c981c1802bb4  experiments/002_full_5000/results.json
c687653de340cd97be7ed75c3cb050c91559bf2896ffb1297b7193af9666d384  experiments/002_full_5000/run.log
```

## Third-experiment preparation on 2026-10-05

At the user's request, the unused 500-image working configuration was replaced
by [003_corrected_full](003_corrected_full/README.md), and the root launcher
was redirected to it. The intended full run covers all 5,000 annotated COCO
val2017 images. This records preparation, not a completed full experiment;
no corrected full-run metrics are available yet. Technical preflight or
smoke-test artifacts, when created, are not research-result evidence.

The new protocol addresses the ten reviewed issues in separate code: sparse
COCO class mapping, darkening gamma, fail-fast evaluation, consistent empty
predictions, explicit detector settings, zero-flux AD boundaries, input
validation, protected and richer output, a matched global-brightness control,
and finite/undefined correlation handling. Baseline, AD, AD+LoG, and the
brightness control share each corrupted image. Noise is deterministic per
image and condition. The control matches mean BGR code-value intensity, not
physical luminance or all contrast properties.

Each future inference run gets a unique directory with source snapshots,
manifest/checksums, status, raw detections, per-image diagnostics, incremental
metrics, and a log. `results.json` is written only after success. Raw
detections and default smoke directories are ignored by Git; full aggregate
artifacts remain eligible for publication. The existing setup and license
were not changed, and historical 001/002 code and artifacts were not edited.

The [analysis plan](003_corrected_full/README.md#analysis-plan-and-limits)
specifies within-run AP comparisons against AD and the brightness control,
including clean-image regressions. Paired image bootstrap and repeated noise
seeds remain follow-up analysis/experiments, not completed evidence. The new
protocol cannot turn the archived defective scores into valid benchmarks or
establish the hypothesis before its own results are collected and analyzed.

### Technical verification on 2026-10-06 UTC

The full `--check-only` preflight read and hashed all 5,000 annotated images,
validated their dimensions, loaded the local detector without inference, and
verified its mapping to all 80 categories. Input fingerprints were:

```text
e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f  shared/coco/annotations/instances_val2017.json
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1  shared/yolo11n.pt
```

Two separate three-image smoke runs completed all six conditions and four
methods, with 72 detector calls in each. The first exposed the installed
Ultralytics 8.4.171 deprecation warning for `half`; the final configuration
uses the explicit FP32 option `quantize=32` instead. No research parameter
was tuned against the smoke scores. The second run completed without warnings
and confirmed the CPU/FP32 backend. Its local, ignored directory is
`003_corrected_full/runs/smoke_20261006T030812Z_b5491332/`; it is not a
published full-run result. It evaluated image IDs 50811, 377497, and 449996.
All 24 prediction streams contain exactly that image sequence, and all 60
artifact fingerprints recorded in its summary were verified.

The final test suite passed all 61 tests locally. A separate export of
publishable files passed 60 tests and skipped only the unavailable local
notebook test. Tests cover numerical preprocessing, sparse category mapping,
empty predictions, evaluation failures, input integrity, output protection,
seed pairing, and artifact persistence. Published-file privacy/link checks
and `git diff --check` passed. Historical artifact fingerprints were retained.
No full corrected run, bootstrap analysis, commit, or push was performed
during this preparation.
