# Experiment Results

- Experiment: `01/runs/pilot_5`
- Recorded image count: 5
- Random seed: 42

[Machine-readable summary](summary.json) | [Source artifact](results.json) | [Protocol and provenance](README.md)

Scores are fractions, not percentages. Tables round to six decimal places;
the JSON retains the recorded precision. `Not run` means the method was
not evaluated; `N/A` means a recorded metric is undefined.

Legacy results retain the incorrect class mapping and brightening
`low_light` transform. These are not corrected COCO benchmark scores.

A shared display format does not make legacy and corrected experiments
directly comparable. Correlation alone does not establish detection benefit.

## mAP at IoU 0.50:0.95

| Condition | Baseline | AD only | AD + LoG | AD + brightness control |
| --- | ---: | ---: | ---: | ---: |
| `clean` | 0.040000 | 0.040000 | 0.040000 | Not run |
| `gaussian_noise` | 0.050495 | 0.020000 | 0.020000 | Not run |
| `blur` | 0.040000 | 0.000000 | 0.000000 | Not run |
| `low_light` | 0.040000 | 0.040000 | 0.040000 | Not run |
| `haze` | 0.040000 | 0.030000 | 0.020000 | Not run |
| `multifactor` | 0.095050 | 0.030000 | 0.020000 | Not run |

## mAP at IoU 0.50

| Condition | Baseline | AD only | AD + LoG | AD + brightness control |
| --- | ---: | ---: | ---: | ---: |
| `clean` | 0.100000 | 0.100000 | 0.100000 | Not run |
| `gaussian_noise` | 0.050495 | 0.100000 | 0.100000 | Not run |
| `blur` | 0.100000 | 0.000000 | 0.000000 | Not run |
| `low_light` | 0.100000 | 0.100000 | 0.100000 | Not run |
| `haze` | 0.100000 | 0.100000 | 0.100000 | Not run |
| `multifactor` | 0.100000 | 0.100000 | 0.100000 | Not run |

## LoG/Gradient Correlation

| Condition | Mean | Standard deviation |
| --- | ---: | ---: |
| `clean` | 0.334006 | 0.038359 |
| `gaussian_noise` | 0.303330 | 0.039605 |
| `blur` | 0.313374 | 0.050339 |
| `low_light` | 0.320331 | 0.023972 |
| `haze` | 0.352232 | 0.030793 |
| `multifactor` | 0.331450 | 0.022524 |

## Common Parameters

| Parameter | Recorded value |
| --- | ---: |
| `AD_KAPPA` | 30 |
| `AD_NITER` | 5 |
| `AD_GAMMA` | 0.15 |
| `LOG_SIGMA` | 1.5 |
| `LOG_BETA` | 0.3 |
| `N_IMAGES` | 5 |
| `RANDOM_SEED` | 42 |

These are the parameters shared by the compact result schema, not the
complete inference/degradation configuration. See the protocol and source
artifact for additional metadata and known provenance gaps.
