# Archived experiment results

These files preserve the recorded AD / AD+LoG / baseline experiment outputs.
The full run has known methodology issues described below; its scores should
not be treated as a validated COCO benchmark.

## Runs

| Output | Images | Description |
| --- | ---: | --- |
| [results.json](results.json) | 5 | Preliminary run; image count and parameters are recorded in the JSON. |
| [full_5000.AfmLxA/results.json](full_5000.AfmLxA/results.json) | 5,000 | Full COCO val2017 run with a completed [run.log](full_5000.AfmLxA/run.log). |

Both JSON files contain `map_results` (mAP at IoU 0.50:0.95 and 0.50),
`corr_summary` (mean and standard deviation of the LoG/gradient correlation),
and `params`. Scores are fractions, not percentages.

## Full run metadata

- Started: `2026-10-02T23:17:48.941074+00:00`, as recorded in `RUN_INFO`.
- Completed: `RUN_FINISHED elapsed_seconds=7169.4`.
- Detector: frozen, fused YOLO11n.
- Environment recorded by the log: PyTorch `2.14.1+cu130`, CUDA unavailable.
- Conditions: `clean`, `gaussian_noise`, `blur`, `low_light`, `haze`, `multifactor`.
- Methods: `baseline`, `ad_only`, `ad_log`.
- Seed: `42`; AD: kappa `30`, iterations `5`, gamma `0.15`;
  LoG: sigma `1.5`, beta `0.3`.

## Interpretation and reproducibility limits

The full run explicitly records these known limitations in `RUN_INFO`:

- YOLO-to-COCO class mapping was not corrected. The source uses `cls + 1`,
  which does not account for gaps in COCO category IDs and affects mAP evaluation.
- The `low_light` transform uses gamma `0.5`, which brightens images. This
  also affects `multifactor`, which includes the same transform.

The current [experiment script](../experiment/run_experiment.py) defaults to
500 images and writes to `results/results.json`. The archived runs used 5 and
5,000 images, respectively. The exact launch command, a source snapshot, full
dependency versions, and model checksum were not saved with these outputs;
the log and JSON parameters are the available run metadata. A default rerun
would overwrite the preliminary result, so use a separate output directory
for future runs. Preserve these files when recording corrected experiments.
