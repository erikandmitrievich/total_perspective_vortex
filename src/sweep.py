"""
The 109-subject, six-experiment sweep.

Routes through ``fit_pipeline`` and ``score_stream`` rather than ``train``
and ``predict``: a sweep cell is then the same number as a CLI run, minus
the disk round trip.
"""

import numpy as np

from src.data import DEFAULT, PreprocConfig, load_dataset
from src.experiments import EXPERIMENTS, SUBJECTS
from src.predict import score_stream
from src.train import FIT_DEFAULT, FitConfig, fit_pipeline


def evaluate(
        subject: int,
        runs: list[int],
        config: PreprocConfig = DEFAULT,
        fit: FitConfig = FIT_DEFAULT,
) -> float:
    """
    Train and score one subject/experiment cell of the sweep.

    Mirrors ``train`` + ``predict`` without touching the disk: the fitted
    estimator and the split stay in memory and are discarded after
    scoring. The CV scores from ``fit_pipeline`` are dropped — the sweep
    reports held-out accuracy only.

    Parameters
    ----------
    subject : int
        Subject number, 1-109.
    runs : list of int
        One entry of ``EXPERIMENTS``.
    config : PreprocConfig
        Preprocessing parameters.
    fit : FitConfig
        Split and estimator parameters.

    Returns
    -------
    acc : float
        Held-out accuracy for this cell.
    """
    X, y, _ = load_dataset(subject, runs, config)
    clf, _, test_idx, scores = fit_pipeline(X, y, fit)
    return score_stream(clf, X, y, test_idx, verbose=False)


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

    for exp, (runs, label) in enumerate(EXPERIMENTS):
        accs = []
        for subject in SUBJECTS:
            try:
                acc = evaluate(subject, runs)
            except Exception as e:
                failures.append((exp, subject, repr(e)))
                print(f"experiment {exp}: subject {subject:03d}: FAILED ({e!r})")
                continue
            accs.append(acc)
            print(f"experiment {exp}: subject {subject:03d}: "
                  f"accuracy = {acc:.4f}  ({label})")

        group_means.append(float(np.mean(accs)) if accs else float("nan"))
        group_counts.append(len(accs))

    print("\nMean accuracy of the six different experiments:")
    for exp, (m, n, (_, label)) in enumerate(
            zip(group_means, group_counts, EXPERIMENTS)):
        print(f"experiment {exp}:      accuracy = {m:.4f}  (n={n}, {label})")
    print(f"\nMean accuracy of {len(EXPERIMENTS)} experiments: "
          f"{np.mean(group_means):.4f}")

    if failures:
        print(f"\n{len(failures)} cells failed:")
        for exp, subject, err in failures:
            print(f"  experiment {exp}, subject {subject:03d}: {err}")

    return group_means
