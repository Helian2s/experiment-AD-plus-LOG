# Preliminary experiment on 5 images

This folder preserves the historical [results.json](results.json) for a
5-image preliminary run. No original run log or exact execution source
snapshot was saved. The result's observed modification time was
2026-10-02 23:14:19 UTC; that is not a verified execution timestamp.

## Result files

- [RESULTS.md](RESULTS.md): generated tables using the same layout as the
  other runs.
- [summary.json](summary.json): derived compact view in the common schema.
- [run_config.json](run_config.json): retrospective parameter record derived
  from the archived result, not an original execution manifest.
- [results.json](results.json): original machine-readable result, preserved
  byte-for-byte with its `map_results`, `corr_summary`, and `params` schema.

The brightness control was not run in this experiment. Its report column is
marked `Not run`; no missing measurements are replaced by zero. Formatting
does not correct the historical protocol or recalculate its scores.

## Code provenance

The shared [run_experiment.py](../../run_experiment.py) belongs to code version
`01`. It was reconstructed during the 2026-10-05 reorganization;
the pilot and full reconstructions were consolidated on 2026-10-06 because
only their image counts differed. `--images 5` selects the pilot configuration
for a new execution. Processing and evaluation algorithms were not corrected.
This is not the original execution snapshot and does not establish exact
reproduction of the archived run.

## Recorded configuration

- Random seed: `42`.
- AD: kappa `30`, iterations `5`, time step `0.15`.
- LoG: sigma `1.5`, modulation beta `0.3`.
- Six conditions: `clean`, `gaussian_noise`, `blur`, `low_light`, `haze`, `multifactor`.
- Three methods: `baseline`, `ad_only`, `ad_log`.

The legacy implementation has incorrect YOLO-to-COCO class mapping and
brightens images in the condition named `low_light`. The stored metrics are
preliminary and are not a validated benchmark.

The script refuses to reuse this archived run directory. Follow the
[version instructions](../../README.md#additional-runs) to create another run
of the same code, or a new version for algorithm changes. Shared data and
weights live under `shared/`; the shared Python environment is `.venv/`
at the repository root.
