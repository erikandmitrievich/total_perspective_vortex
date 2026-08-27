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

Held-out accuracy, `seed=42`:

| Experiment | Accuracy |
|------------|----------|
| 0          | 0.5806   |
| 1          | 0.6147   |
| 2          | 0.7254   |
| 3          | 0.6390   |
| 4          | 0.5599   |
| 5          | 0.6890   |
| **mean**   | **0.6348** |

Above the 60% mean the subject requires.

## Layout

```
.
├── mybci.py                          CLI: sweep / train / predict
├── visualise.py                      before/after PSD figure
├── requirements.txt
├── src/
│   ├── data.py                       load, filter, epoch
│   ├── csp.py                        MyCSP - the from-scratch transformer
│   ├── train.py                      pipeline, cross-validation, fit, persistence
│   ├── predict.py                    stream replay and scoring
│   ├── sweep.py                      109 subjects x 6 experiments
│   └── experiments.py                run groups
└── docs/
    ├── csp.org                       CSP derivation and verification notes
    ├── eegmmidb.org                  dataset layout
    └── total_perspective_vortex.org  the spec
```

`fit_pipeline` and `score_stream` are pure — arrays in, numbers out. `train`, `predict` and `evaluate` are shells over them.
