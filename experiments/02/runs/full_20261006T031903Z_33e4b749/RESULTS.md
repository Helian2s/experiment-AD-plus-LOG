# Experiment Results

- Experiment: `02/runs/full_20261006T031903Z_33e4b749`
- Recorded image count: 5000
- Random seed: 42

[Machine-readable summary](summary.json) | [Source artifact](results.json) | [Protocol and provenance](../../README.md)

Scores are fractions, not percentages. Tables round to six decimal places;
the JSON retains the recorded precision. `Not run` means the method was
not evaluated; `N/A` means a recorded metric is undefined.

Corrected evaluation with a matched-brightness control. This is one
seed; confidence intervals and repeated-seed analysis are not included.

A shared display format does not make legacy and corrected experiments
directly comparable. Correlation alone does not establish detection benefit.

## mAP at IoU 0.50:0.95

| Condition | Baseline | AD only | AD + LoG | AD + brightness control |
| --- | ---: | ---: | ---: | ---: |
| `clean` | 0.386713 | 0.372117 | 0.368681 | 0.372191 |
| `gaussian_noise` | 0.246103 | 0.280893 | 0.277958 | 0.279978 |
| `blur` | 0.367129 | 0.344672 | 0.337860 | 0.344518 |
| `low_light` | 0.365776 | 0.345462 | 0.343618 | 0.344951 |
| `haze` | 0.354219 | 0.333108 | 0.317136 | 0.333052 |
| `multifactor` | 0.200841 | 0.216406 | 0.197491 | 0.216378 |

## mAP at IoU 0.50

| Condition | Baseline | AD only | AD + LoG | AD + brightness control |
| --- | ---: | ---: | ---: | ---: |
| `clean` | 0.541004 | 0.521632 | 0.518038 | 0.521837 |
| `gaussian_noise` | 0.361810 | 0.408042 | 0.404420 | 0.406756 |
| `blur` | 0.516677 | 0.487697 | 0.479560 | 0.487433 |
| `low_light` | 0.516625 | 0.489766 | 0.487355 | 0.488835 |
| `haze` | 0.500641 | 0.471922 | 0.452010 | 0.471964 |
| `multifactor` | 0.296358 | 0.319962 | 0.292805 | 0.319508 |

## LoG/Gradient Correlation

| Condition | Mean | Standard deviation |
| --- | ---: | ---: |
| `clean` | 0.305327 | 0.046319 |
| `gaussian_noise` | 0.259827 | 0.046450 |
| `blur` | 0.294221 | 0.043961 |
| `low_light` | 0.330460 | 0.047116 |
| `haze` | 0.316070 | 0.040523 |
| `multifactor` | 0.326375 | 0.043485 |

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
