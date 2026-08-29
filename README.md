# Total Perspective Vortex

Motor-imagery classification from EEG: a from-scratch Common Spatial Patterns transformer feeding an LDA classifier, wired into a scikit-learn `Pipeline` and scored on the PhysioNet EEGMMIDB.

## What it does

Given a subject's 64-channel recording, decide which of two movements they performed or imagined: left fist vs right fist, or both fists vs both feet.

1. **Load** — concatenate the runs of one task group (`mne.datasets.eegbci`).
2. **Preprocess** — 7–30 Hz band-pass, `standard_1005` montage.
3. **Epoch** — `T1`/`T2` from annotations, 1–2 s after onset.
4. **Reduce** — `MyCSP`: solves $S_0 w = \mu (S_0 + S_1) w$, keeps the extreme filters, returns log band power.
5. **Classify** — linear discriminant analysis.
6. **Replay** — held-out epochs predicted one at a time, measuring per-chunk latency against the 2 s bound.

CSP is implemented from scratch in `src/csp.py`; its derivation is in [`docs/csp.org`](docs/csp.org).

## Install

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The dataset downloads on first use into MNE's cache; it is not vendored.

## Usage

```sh
# full sweep: 109 subjects x 6 experiments
python mybci.py

# cross-validate and persist a model for one subject and task group
python mybci.py 4 14 train

# replay that model's held-out epochs as a stream
python mybci.py 4 14 predict

# PSD before and after preprocessing
python visualise.py 1 6
```

`train` and `predict` take a run number and expand it to its group; a single run is not a self-contained binary problem. The model is written to `models/S{subject}_R{runs}.joblib` with the preprocessing config and held-out indices, so `predict` scores the same partition under the same pipeline.

## Experiments

| # | Runs               | Task                                |
|---|--------------------|-------------------------------------|
| 0 | 3, 7, 11           | left fist vs right fist, executed   |
| 1 | 4, 8, 12           | left fist vs right fist, imagined   |
| 2 | 5, 9, 13           | both fists vs both feet, executed   |
| 3 | 6, 10, 14          | both fists vs both feet, imagined   |
| 4 | 3, 4, 7, 8, 11, 12 | left fist vs right fist, pooled     |
| 5 | 5, 6, 9, 10, 13, 14| both fists vs both feet, pooled     |

Experiments 4 and 5 pool an executed group with its imagined counterpart: T1/T2 mean the same movements in both, so the labels stay consistent and the epoch count doubles. They are reachable from the sweep only, not the CLI.

## Results

Held-out accuracy, mean over all 109 subjects. `seed 0` is the default (`FitConfig.seed`); the spread is over seeds 0–4.

| # | Task                 | seed 0     | mean       | min        | max        |
|---|----------------------|------------|------------|------------|------------|
| 0 | L/R fist, executed   | 0.5766     | 0.5752     | 0.5678     | 0.5794     |
| 1 | L/R fist, imagined   | 0.5604     | 0.5696     | 0.5604     | 0.5729     |
| 2 | fists/feet, executed | 0.7229     | 0.7286     | 0.7202     | 0.7363     |
| 3 | fists/feet, imagined | 0.6373     | 0.6391     | 0.6368     | 0.6425     |
| 4 | L/R fist, pooled     | 0.5932     | 0.5866     | 0.5816     | 0.5932     |
| 5 | fists/feet, pooled   | 0.6875     | 0.6879     | 0.6865     | 0.6891     |
|   | **mean**             | **0.6296** | **0.6312** | **0.6296** | **0.6335** |

Above the 60% mean the spec requires, at every seed tried. The last row's min and max are over per-seed overall means, not over the columns above them. Raw sweep output in [`results/`](results/).

## Layout

```
.
├── mybci.py                CLI: sweep / train / predict
├── visualise.py            before/after PSD figure
├── requirements.txt
├── LICENSE
├── src/
│   ├── data.py             load, filter, epoch
│   ├── csp.py              MyCSP — the from-scratch transformer
│   ├── train.py            pipeline, cross-validation, fit, persistence
│   ├── predict.py          stream replay and scoring
│   ├── sweep.py            full sweep over subjects × experiments
│   └── experiments.py      run groups
├── results/
│   └── seed_NN.log         full sweep output, one file per seed
├── figs/
│   └── S001_R6_psd.png     sample output of visualise.py
└── docs/
    ├── subject.org         the spec
    ├── design.org          module-by-module walkthrough
    └── csp.org             CSP derivation
```

`fit_pipeline` and `score_stream` are pure — arrays in, numbers out. `train`, `predict` and `evaluate` are shells over them.
