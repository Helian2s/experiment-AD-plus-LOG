# Full experiment on 5000 images

This folder preserves the historical [results.json](results.json) and
[run.log](run.log) from the completed COCO val2017 run. Both files were moved
without changing their bytes. For publication, the log's local project-root
prefix was subsequently replaced with `<PROJECT_ROOT>` in two places; all
other log bytes and the JSON remain unchanged. Their previous location was
`results/full_5000.AfmLxA/`; that historical suffix is retained in the log.
The [trace](../TRACE.md#public-repository-cleanup-on-2026-10-05) documents the
privacy edit and current fingerprints.

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

[run_experiment.py](run_experiment.py) is a reconstruction created during the
2026-10-05 reorganization from the then-current implementation, using the
recorded `N_IMAGES = 5000`. Paths and overwrite protection were updated for
this folder; processing and evaluation algorithms remain unchanged. The exact
original execution source, launch command, and complete environment were
not saved, so this file is not an original execution snapshot.

## Known limitations

The log explicitly records incorrect `cls + 1` YOLO-to-COCO class mapping
and the gamma 0.5 transform that brightens images under the `low_light` label.
The latter also affects `multifactor`. These are uncorrected preliminary
scores, not valid COCO benchmark measurements or proof of the research
hypothesis. Raw detections were not preserved, so the saved aggregate scores
cannot simply be reevaluated with a corrected class mapping.

The script refuses to overwrite the archived result. Follow the
[experiment index](../README.md#running-and-adding-experiments) to create a
new experiment for a corrected run. Shared data and weights remain under
`shared/`; the shared Python environment is `.venv/` at the repository root.
