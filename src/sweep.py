"""
The 109-subject, six-experiment sweep.

Routes through ``fit_pipeline`` and ``score_stream`` rather than ``train``
and ``predict``, so no cell touches the disk.

A cell is not the same number as a CLI run: ``predict`` scores one holdout
at ``FitConfig.seed``, a cell averages ``N_REPEATS`` holdouts drawn from
seeds derived from it. The two answer different questions — one number for
one split, versus an estimate of the expected accuracy over splits.
"""

from dataclasses import replace

import numpy as np

from src.data import DEFAULT, PreprocConfig, load_dataset
from src.experiments import EXPERIMENTS, SUBJECTS
from src.predict import score_stream
from src.train import FIT_DEFAULT, FitConfig, fit_pipeline


N_REPEATS = 5
"""
Holdout draws averaged per sweep cell.
"""


def evaluate(
        subject: int,
        runs: list[int],
        config: PreprocConfig = DEFAULT,
        fit: FitConfig = FIT_DEFAULT,
        *,
        n_repeats: int = N_REPEATS,
) -> float:
    """
    Train and score one subject/experiment cell of the sweep.

    Fits ``n_repeats`` times on ``n_repeats`` stratified holdouts of the
    same epochs and averages the held-out accuracies. Nothing is
    persisted: each estimator is discarded after scoring, and the CV
    scores from ``fit_pipeline`` are never requested.

    ``load_dataset`` runs once for all repeats — the EDF read and the
    band-pass dominate the cost, so the repeats are close to free.

    Parameters
    ----------
    subject : int
        Subject number, 1-109.
    runs : list of int
        One entry of ``EXPERIMENTS``.
    config : PreprocConfig
        Preprocessing parameters.
    fit : FitConfig
        Split and estimator parameters. ``fit.seed`` is not the split seed
        here: it seeds the derivation of one seed per repeat, so every cell
        draws its own splits and the whole sweep stays reproducible from
        one number.
    n_repeats : int
        Holdout draws to average over.

    Returns
    -------
    acc : float
        Mean held-out accuracy across the repeats.
    """
    X, y, _ = load_dataset(subject, runs, config)

    accs = []
    for k in range(n_repeats):
        seed = (fit.seed * 100_003 + subject * 101 + k) % (2**31 - 1)
        clf, _, test_idx, _ = fit_pipeline(X, y, replace(fit, seed=seed),
                                           cross_validate=False)
        accs.append(score_stream(clf, X, y, test_idx, verbose=False))

    return float(np.mean(accs))


def run_full_sweep() -> list[float]:
    """
    Score every subject on every experiment and report the group means.

    Prints one line per cell as it goes, then the six experiment means and
    their mean, in the format the subject specifies.

    Returns
    -------
    group_means : list of float
        One mean per entry of ``EXPERIMENTS``, in order. ``nan`` for an
        experiment where every subject failed.
    """
    group_means, group_counts, failures = [], [], []

    print(f"sweep: seed={FIT_DEFAULT.seed}, n_repeats={N_REPEATS}, "
          f"{len(SUBJECTS)} subjects x {len(EXPERIMENTS)} experiments\n")
    for exp, (runs, label) in enumerate(EXPERIMENTS):
        accs = []
        for subject in SUBJECTS:
            try:
                acc = evaluate(subject, runs)
            except Exception as e:
                failures.append((exp, subject, repr(e)))
                print(f"experiment {exp}: subject {subject:03d}: "
                      f"FAILED ({e!r})")
                continue
            accs.append(acc)
            print(f"experiment {exp}: subject {subject:03d}: "
                  f"accuracy = {acc:.4f}  ({label})")

        group_means.append(float(np.mean(accs)) if accs else float("nan"))
        group_counts.append(len(accs))

    print(f"\nMean accuracy of the six different experiments "
          f"for all {len(SUBJECTS)} subjects:")
    for exp, (m, n, (_, label)) in enumerate(
            zip(group_means, group_counts, EXPERIMENTS)):
        print(f"experiment {exp}:      accuracy = {m:.4f}  (n={n}, {label})")
    print(f"\nMean accuracy of {len(EXPERIMENTS)} experiments: "
          f"{np.mean(group_means):.4f}")

    if failures:
        print(f"\n{len(failures)} cells failed:")
        for exp, subject, err in failures:
            print(f"  experiment {exp}, subject {subject:03d}: {err}")

    if not any(group_counts):
        raise RuntimeError(
            f"every cell failed ({len(failures)} of "
            f"{len(SUBJECTS) * len(EXPERIMENTS)}); is the dataset available?"
        )

    return group_means
