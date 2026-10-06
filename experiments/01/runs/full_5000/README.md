# Full experiment on 5000 images

This folder preserves the historical [results.json](results.json) and
[run.log](run.log) from the completed COCO val2017 run. Both files were moved
without changing their bytes. For publication, the log's local project-root
prefix was subsequently replaced with `<PROJECT_ROOT>` in two places; all
other log bytes and the JSON remain unchanged. Their previous location was
`results/full_5000.AfmLxA/`; that historical suffix is retained in the log.
The [trace](../../../TRACE.md#public-repository-cleanup-on-2026-10-05) documents the
privacy edit and current fingerprints.

## Result files

- [RESULTS.md](RESULTS.md): generated tables using the same layout as the
  other runs.
- [summary.json](summary.json): derived compact view in the common schema.
- [run_config.json](run_config.json): retrospective parameter record derived
  from the archived result, not an original execution manifest.
- [results.json](results.json): original machine-readable result, preserved
  byte-for-byte with its `map_results`, `corr_summary`, and `params` schema.
- [run.log](run.log): the privacy-sanitized historical execution log.

The brightness control was not run in this experiment. Its report column is
marked `Not run`; no missing measurements are replaced by zero. Formatting
does not correct the historical protocol or recalculate its scores.

## Recorded run

- Started: `2026-10-02T23:17:48.941074+00:00`.
- Completed: `RUN_FINISHED elapsed_seconds=7169.4`.
- Images: `5000`, selected from `5000` available; random seed `42`.
- Detector: frozen, fused YOLO11n.
- Recorded environment: PyTorch `2.14.1+cu130`, CUDA unavailable.
- AD: kappa `30`, iterations `5`, time step `0.15`.
- LoG: sigma `1.5`, modulation beta `0.3`.
- Six conditions: `clean`, `gaussian_noise`, `blur`, `low_light`, `haze`, `multifactor`.
- Three methods: `baseline`, `ad_only`, `ad_log`.

## Code provenance

The shared [run_experiment.py](../../run_experiment.py) belongs to code version
`01`. It was reconstructed during the 2026-10-05 reorganization;
the pilot and full reconstructions were consolidated on 2026-10-06 because
only their image counts differed. `--images 5000` selects the full configuration
for a new execution. Processing and evaluation algorithms remain unchanged.
The exact original execution source, launch command, and complete environment
were not saved, so this file is not an original execution snapshot.

## Known limitations

The log explicitly records incorrect `cls + 1` YOLO-to-COCO class mapping
and the gamma 0.5 transform that brightens images under the `low_light` label.
The latter also affects `multifactor`. These are uncorrected preliminary
scores, not valid COCO benchmark measurements or proof of the research
hypothesis. Raw detections were not preserved, so the saved aggregate scores
cannot simply be reevaluated with a corrected class mapping.

The script refuses to reuse this archived run directory. Follow the
[version instructions](../../README.md#additional-runs) to create another run
of the same code. For corrected evaluation use experiment 02, not this legacy
protocol. Shared data and weights remain under `shared/`; the shared Python
environment is `.venv/` at the repository root.
