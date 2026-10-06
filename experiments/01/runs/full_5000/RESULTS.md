# Experiment Results

- Experiment: `01/runs/full_5000`
- Recorded image count: 5000
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
| `clean` | 0.054642 | 0.052697 | 0.052340 | Not run |
| `gaussian_noise` | 0.035968 | 0.040257 | 0.040256 | Not run |
| `blur` | 0.051988 | 0.049382 | 0.048653 | Not run |
| `low_light` | 0.052806 | 0.051334 | 0.050811 | Not run |
| `haze` | 0.048344 | 0.045544 | 0.045086 | Not run |
| `multifactor` | 0.029195 | 0.033943 | 0.035100 | Not run |

## mAP at IoU 0.50

| Condition | Baseline | AD only | AD + LoG | AD + brightness control |
| --- | ---: | ---: | ---: | ---: |
| `clean` | 0.072940 | 0.070224 | 0.070053 | Not run |
| `gaussian_noise` | 0.048826 | 0.054571 | 0.054469 | Not run |
| `blur` | 0.069438 | 0.066515 | 0.065808 | Not run |
| `low_light` | 0.070897 | 0.069274 | 0.068478 | Not run |
| `haze` | 0.064830 | 0.061053 | 0.060939 | Not run |
| `multifactor` | 0.039844 | 0.045878 | 0.047611 | Not run |

## LoG/Gradient Correlation

| Condition | Mean | Standard deviation |
| --- | ---: | ---: |
| `clean` | 0.305258 | 0.046292 |
| `gaussian_noise` | 0.259847 | 0.046441 |
| `blur` | 0.294049 | 0.043862 |
| `low_light` | 0.298363 | 0.045166 |
| `haze` | 0.315819 | 0.040457 |
| `multifactor` | 0.306831 | 0.040265 |

## Common Parameters

| Parameter | Recorded value |
| --- | ---: |
| `AD_KAPPA` | 30 |
| `AD_NITER` | 5 |
| `AD_GAMMA` | 0.15 |
| `LOG_SIGMA` | 1.5 |
| `LOG_BETA` | 0.3 |
| `N_IMAGES` | 5000 |
| `RANDOM_SEED` | 42 |

These are the parameters shared by the compact result schema, not the
complete inference/degradation configuration. See the protocol and source
artifact for additional metadata and known provenance gaps.
