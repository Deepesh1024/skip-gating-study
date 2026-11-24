# Adaptive skip gating under varying skip reliability — code

Code, data split and result files for the paper *"Skip Reliance Is Set by Training, Not by the Gate: A Controlled Study of Adaptive Skip Gating"* (anonymous submission).

All result files used for the paper's tables, figures and appendix are included, so every number can be regenerated **without a GPU** (see *Regenerate figures and tables*). Retraining requires the dataset and a GPU.

## Contents

| Path | What it is |
|---|---|
| `model.py` | Four-level U-Net with interchangeable skip-fusion operators: `concat`, `static`, `arf` (input-dependent gate), `arf` without the difference input, `noskip` |
| `data.py` | LGG FLAIR loader: FLAIR channel, per-image z-score over nonzero pixels, binarized masks |
| `train.py` | Training (clean or skip-corruption regime) |
| `metrics.py` | Per-patient Dice from summed TP/FP/FN |
| `run_grid.py` | Runs the 24 main training runs (4 operators x 2 regimes x 3 seeds) |
| `eval.py` | One-time test evaluation (ID and OOD); refuses to run unless all runs are complete |
| `analyze.py` | Pre-specified statistics (patient-clustered bootstrap, interaction test, Holm correction), Fig. 1 |
| `zero_skip_val.py` | Post-hoc zeroing control (validation set) |
| `gate_probe.py` | Post-hoc probe: unseen skip perturbations, gate response, gate regret (validation set) |
| `subset_probe.py` | Post-hoc probe: removal of every subset of skip levels, pixel-level gate change (validation set) |
| `make_fig2.py`, `make_fig3.py`, `make_appendix.py` | Paper figures 2–3 and all appendix tables, generated from the result files |
| `make_split_final.py`, `split.json`, `split.sha256` | Patient-level split (held-out site FG = OOD test), fixed before training |
| `phase0.py` | Sanity checks of loader, split and operators before training |
| `audit/manifest.csv`, `audit/manifest.sha256` | Slice-level manifest of the dataset (path, patient, site, empty mask) |
| `results/*.json` | Test-set results for all 24 runs (per-patient Dice, corruptions, skip shuffling, gate statistics) |
| `gate_probe_val.json`, `subset_probe_val.json`, `zero_skip_val.json`, `analysis.json` | Outputs of the probes and the pre-specified analysis |
| `ns0.txt`–`ns2.txt` | Training logs of the no-skip baseline (seeds 0–2) |
| `logs/` | Console output of evaluation, analysis and probes |

## Setup

```bash
pip install -r requirements.txt
```

**Data.** Download the LGG brain-MRI segmentation dataset (Buda et al.; Kaggle `mateuszbuda/lgg-mri-segmentation`, CC BY-NC-SA 4.0) and place the `kaggle_3m/` folder at `data/lgg/kaggle_3m/`. The data are not redistributed here. `make_split_final.py` verifies the manifest checksum before building a split.

## Reproduce the experiments (GPU)

```bash
python phase0.py                      # sanity checks
python run_grid.py 0,1,2 3            # 24 main runs, 80 epochs, 3 runs concurrently -> runs/
for s in 0 1 2; do python train.py --variant noskip --regime clean --seed $s --epochs 80; done
python eval.py                        # one-time test evaluation -> results/
python analyze.py                     # pre-specified statistics, Fig. 1
python zero_skip_val.py               # post-hoc zeroing control (validation)
python gate_probe.py                  # post-hoc probe 1 (validation)
python subset_probe.py                # post-hoc probe 2 (validation)
```

Notes:
- `train.py` defaults to 60 epochs; the paper uses 80 (as set in `run_grid.py`).
- The hypotheses, decision threshold and split were fixed before the test sets were evaluated; each test set was evaluated once. All probes use the validation set only.
- Runs are not bit-reproducible (CUDA nondeterminism).
- Hardware used: one NVIDIA L40S, three runs concurrently, 0.6–1.0 h per run.

## Regenerate figures and tables (CPU, from the included result files)

```bash
python make_fig2.py                   # Fig. 2 (skip reliance index)
python make_fig3.py                   # Fig. 3 (level-subset removal)
python make_appendix.py               # all appendix tables -> appendix.tex
```

Smoke tests of the probe plumbing (random weights, synthetic data, no dataset needed):

```bash
python subset_probe.py --smoke 1 --device cpu
python gate_probe.py --smoke 1 --device cpu --seeds 0 --variants arf --regimes clean
```

## License of the data

The dataset is distributed by its authors under CC BY-NC-SA 4.0. Cite Buda et al. (2019) and Mazurowski et al. (2017) when using it.
