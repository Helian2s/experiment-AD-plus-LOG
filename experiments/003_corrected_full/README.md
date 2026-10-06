# Experiment 003: corrected full COCO val2017 evaluation

Status: preparation for a new full run, not a completed research result.
This replaces the unused `003_current_500` configuration; no 500-image result
was archived. Experiments 001 and 002 retain their original artifacts and
reconstructed legacy code.

## Question and fixed protocol

For a frozen YOLO11n detector, does AD followed by spatial LoG modulation
improve object-detection AP over AD alone and over a matched global-brightness
control under the specified synthetic degradations?

- Data: every image listed in `instances_val2017.json`, expected to be all
  5,000 COCO val2017 images. All methods use the same ordered image IDs.
- Conditions: clean, Gaussian noise (sigma 25), Gaussian blur (5 by 5),
  low light (gamma 2.0), synthetic haze, and noise then low light then haze.
  Haze uses beta 1.0, atmospheric light 0.9, and a left-to-right synthetic
  depth ramp from 0.1 to 1.0; it is not a physical depth estimate.
- Methods: `baseline`, `ad_only`, `ad_log`, `ad_brightness_control`.
- AD: kappa 30, 5 iterations, time step 0.15, zero-flux boundaries.
- LoG: sigma 1.5, absolute normalized map, modulation beta 0.3.
- Default seed: 42. A deterministic seed is derived for each image and
  condition, so methods receive the same corrupted input and selection/order
  changes do not change an individual image's noise realization.
- Inference uses explicit image size, confidence threshold, NMS IoU,
  detection limit, device, and precision settings recorded in the manifest.
  The default device is CPU; weights are not trained or updated.

Detector settings are `imgsz=640`, `conf=0.001`, `iou=0.7`, `max_det=300`,
`rect=True`, `batch=1`, all classes, FP32 (`quantize=32` in the installed
Ultralytics 8.4.171), and no augmentation or class-agnostic
NMS. COCO evaluation separately uses `maxDets=[1,10,100]` and the standard
IoU/recall thresholds and area ranges. Finite zero-area boxes caused by
boundary clipping are retained for evaluation and counted, not silently
removed; negative dimensions and nonfinite detections fail validation.

Freeze these settings before interpreting the full run. Changing them after
seeing its results is a new exploratory experiment, not independent
confirmation. The prediction pipeline uses COCO evaluation but does not
claim exact reproduction of the vendor's published YOLO benchmark.

## Corrections and controls

| Previously identified issue | Protocol for experiment 003 |
| --- | --- |
| YOLO index plus one is not a COCO category ID | Validate the model classes and map them to the sparse COCO annotation IDs. |
| `low_light` brightened images | Use exponent 2.0 in low light and the multifactor chain. |
| Evaluation exceptions became zero scores | Fail the run and preserve its failure status; do not create a successful summary. |
| Empty predictions had a different result type | Evaluate an empty detection set with the same metric schema as other methods. |
| Detector options were implicit | Set and record inference parameters and COCO evaluation settings. |
| AD wrapped across image borders | Use zero-flux boundaries, with no opposite-edge coupling. |
| Unreadable images stayed in evaluation | Validate all selected images before inference and fail on missing, unreadable, or changed inputs. |
| Outputs were insufficient and overwritable | Exclusive run directories, source snapshots, checksums, raw predictions, per-image diagnostics, and incremental metrics. |
| LoG benefit was confounded with brightening | Add a uniform-gain AD control matched to the final AD+LoG image's mean BGR code-value intensity. |
| Correlations could be undefined or overinterpreted | Represent undefined correlations as `null` and record validity counts; use detection metrics to assess benefit. |

The brightness control matches the mean after clipping and uint8 conversion,
within quantization limits; diagnostics record its residual mismatch. This is
mean channel code-value intensity, not physical luminance. A uniform gain
does not match local contrast, the intensity histogram, or saturation patterns.
Therefore a difference against this control tests more than global mean
brightness but does not isolate a unique causal role for LoG edge information.

## Running

Use the shared root `.venv/`, dataset, and weights. From the repository root:

```bash
.venv/bin/python run_experiment.py --check-only
.venv/bin/python run_experiment.py --smoke-test
.venv/bin/python run_experiment.py
```

`--check-only` checks all 5,000 images, dimensions, hashes, weights, and model
class mapping without detector inference, printing a report without creating
a run directory or results. `--smoke-test` evaluates three
deterministically selected images across all six conditions and four methods;
its artifacts are labeled smoke and are not evidence for the research question.
The command without either flag evaluates the full annotated split.

Optional flags include `--seed 42`, `--device cpu`, and `--output-dir PATH`.
An explicit output directory must not already exist. Default output is
`runs/full_<timestamp>_<unique-id>/` or `runs/smoke_<timestamp>_<unique-id>/`
inside this experiment. Each inference run requires a new directory; no
automatic overwrite or resume is implied. The full protocol requires
120,000 detector calls: 5,000 images times six conditions times four methods.

## Artifacts

Each inference run keeps its own evidence under its output directory:

- `source_snapshot/`: the experiment's Python modules and README at launch.
- `manifest.json`: image IDs and hashes, model and annotation checksums,
  source hashes, versions, seed, explicit parameters, and public Git revision.
- `effective_detector.json`: resolved predictor options and backend
  precision/device observed after the first processed image.
- `status.json`: completion/failure state and last recorded progress.
- `detections/<condition>/<method>.jsonl.gz`: per-image raw detections,
  including records for images with no detections.
- `diagnostics/<condition>.jsonl`: per-image correlation and intensity data.
- `metrics/<condition>_<method>.json`: incremental evaluation results, so a
  later evaluation failure does not remove earlier completed metrics.
- `results.json`: aggregate results written only after complete success.
- `run.log`: execution output.

Raw detection archives are preserved locally but ignored by Git by default;
keep them to recalculate AP and perform paired uncertainty analysis without
repeating inference. Default smoke directories are also ignored. Aggregate
full-run artifacts are eligible for publication, not automatically committed.
Never interpret an incomplete or failed output directory as a completed run.

## Preparation checks

Verified on 2026-10-06 UTC, before starting any full corrected run:

- Full input preflight passed for all 5,000 images and 80 model/category names.
- The final three-image smoke test completed all six conditions and four
  methods: 72 detector calls and 24 evaluations, on CPU in FP32, without warnings.
- All 61 tests passed locally. In an export containing only publishable files,
  60 passed and the optional local-notebook test was skipped.
- Smoke-test artifact hashes and image-ID consistency were verified. Historical
  JSON/log fingerprints remain unchanged by this preparation.

These are software/input checks, not evidence that AD or LoG improves detection.
The full 5,000-image experiment has not been run. See the
[trace](../TRACE.md#third-experiment-preparation-on-2026-10-05) for details.

## Analysis plan and limits

Primary comparisons are paired AP differences `ad_log - ad_only` and
`ad_log - ad_brightness_control` for each condition. Also compare each method
with baseline and report degradation or improvement on clean images. Use
COCO AP at IoU 0.50:0.95 as the primary metric; inspect AP50, AP75, object-size
metrics, recall, brightness mismatch, and saturation as supporting diagnostics.
Report all six conditions, not only favorable ones. Scores are fractions;
multiply AP differences by 100 to report percentage-point changes.

After collecting predictions, estimate paired confidence intervals by
resampling image units jointly across methods and recomputing dataset AP.
Do not substitute mean per-image AP for COCO AP. A COCO bootstrap must give
repeated draws unique cloned image and annotation IDs, because COCOeval
deduplicates image IDs. Bootstrap analysis is planned, not supplied by this
runner. One noise seed measures only one realization; repeated noise seeds
are needed to assess corruption randomness separately from image sampling.

LoG/gradient correlations describe the maps, not detection benefit. Neither
low correlation nor a positive point estimate proves the hypothesis. The
same fixed detector, synthetic severity levels, and validation dataset limit
generalization. Follow-up studies should include multiple noise seeds,
severity levels, a spatially shuffled or otherwise appropriate sham control,
another detector, and separate tuning/evaluation data. Do not tune on this
full split and then describe it as an untouched confirmation set.

Experiment 002 used defective class mapping, different gamma behavior, and
implicit detector settings. A higher score in 003 cannot by itself establish
an AD or LoG improvement over 002; only within-003 controlled comparisons
answer this protocol's question.

## References

- [Ultralytics validation options](https://docs.ultralytics.com/modes/val/):
  confidence threshold, NMS settings, and inference configuration.
- [Official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py):
  AP aggregation, image-ID handling, and evaluation parameters.
