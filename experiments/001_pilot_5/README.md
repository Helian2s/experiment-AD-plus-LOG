# Preliminary experiment on 5 images

This folder preserves the historical [results.json](results.json) for a
5-image preliminary run. No original run log or exact execution source
snapshot was saved. The result's observed modification time was
2026-10-02 23:14:19 UTC; that is not a verified execution timestamp.

## Code provenance

[run_experiment.py](run_experiment.py) is a reconstruction created during the
2026-10-05 reorganization from the then-current implementation, using the
recorded `N_IMAGES = 5`. Paths and overwrite protection were updated for this
folder. Processing and evaluation algorithms were not corrected. This is
not the original execution snapshot and does not establish exact reproduction
of the archived run.

## Recorded configuration

- Random seed: `42`.
- AD: kappa `30`, iterations `5`, time step `0.15`.
- LoG: sigma `1.5`, modulation beta `0.3`.
- Six conditions: `clean`, `gaussian_noise`, `blur`, `low_light`, `haze`, `multifactor`.
- Three methods: `baseline`, `ad_only`, `ad_log`.

The legacy implementation has incorrect YOLO-to-COCO class mapping and
brightens images in the condition named `low_light`. The stored metrics are
preliminary and are not a validated benchmark.

The script refuses to overwrite the archived result. Follow the
[experiment index](../README.md#running-and-adding-experiments) to create a
new experiment for a rerun. Shared data and weights live under `shared/`;
the shared Python environment is `.venv/` at the repository root.
