from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (StratifiedShuffleSplit, cross_val_score,
                                     train_test_split)

from sklearn.pipeline import Pipeline

from src.csp import MyCSP
from src.data import DEFAULT, PreprocConfig, load_dataset
from dataclasses import dataclass


@dataclass(frozen=True)
class FitConfig:
    """
    Split and estimator parameters shared by training and the sweep.

    Frozen for the same reason as ``PreprocConfig``: the split is part of
    the experiment, so it travels as one value rather than as loose keyword
    arguments that a caller can partially override. Unlike ``PreprocConfig``
    it is not currently written to the model file — ``predict`` needs only
    the persisted ``test_idx``, so the split is reproducible in effect but
    not recorded.

    Attributes
    ----------
    n_components : int
        Spatial filters kept by ``MyCSP``. Must be even.
    test_size : float
        Fraction held out, and the validation fraction inside each CV split.
    n_splits : int
        Number of ``StratifiedShuffleSplit`` resamples.
    seed : int
        Seeds the holdout and the CV both, making a run reproducible.
    """
    n_components: int = 4
    test_size:    float = 0.2
    n_splits:     int = 10
    seed:         int = 42


FIT_DEFAULT = FitConfig()
"""Module-level default fit configuration."""

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def model_path(
        subject: int,
        runs: list[int]
) -> Path:
    """
    Deterministic on-disk location for one fitted model.

    Parameters
    ----------
    subject : int
        Subject number, 1-109.
    runs : list of int
        Run numbers the model was fitted on. Part of the filename, so
        different run groups never collide.

    Returns
    -------
    path : Path
        ``models/S{subject:03d}_R{runs}.joblib``. Not created here.
    """
    return MODELS_DIR / f"S{subject:03d}_R{'-'.join(map(str, runs))}.joblib"


def make_pipeline(
        n_components: int = 4
) -> Pipeline:
    """
    Build the CSP → LDA estimator.

    Single definition of the pipeline: ``train`` and the sweep both build
    it here, so no caller can drift on hyperparameters. ``predict`` does
    not build one at all — it unpickles the estimator fitted by ``train``.

    Parameters
    ----------
    n_components : int, default=4
        Number of spatial filters kept by ``MyCSP``. Must be even.

    Returns
    -------
    clf : Pipeline
        Unfitted estimator, clonable by ``cross_val_score``.
    """
    return Pipeline([
        ("CSP", MyCSP(n_components=n_components)),
        ("LDA", LinearDiscriminantAnalysis()),
    ])


def fit_pipeline(
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.int_],
        fit: FitConfig = FIT_DEFAULT
) -> tuple[Pipeline, npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """
    Cross-validate and fit the pipeline on a stratified training partition.

    A pure function of the arrays: no I/O, no printing, no knowledge of
    subjects or runs. ``train`` and the sweep both route through it, which
    is what makes a CLI result and a sweep entry the same number.

    The holdout is carved out first and never reaches ``cross_val_score``;
    the CV splits partition the training rows only. The returned scores are
    therefore validation scores, and ``test_idx`` is untouched by anything
    here.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Cropped epoch data.
    y : ndarray of int, shape (n_epochs,)
        Labels in {0, 1}, row-aligned with ``X``.
    fit : FitConfig
        Component count, split fractions, resample count and seed.

    Returns
    -------
    clf : Pipeline
        Fitted on ``X[train_idx]`` only.
    train_idx : ndarray of intp
        Training rows, in ``train_test_split`` order.
    test_idx : ndarray of intp
        Held-out rows, sorted ascending so ``score_stream`` replays them in
        recording order rather than shuffle order.
    scores : ndarray of float, shape (n_splits,)
        Per-fold validation accuracies.
    """
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=fit.test_size, stratify=y, random_state=fit.seed
    )

    clf = make_pipeline(fit.n_components)
    cv = StratifiedShuffleSplit(fit.n_splits, test_size=fit.test_size, random_state=fit.seed)
    scores = cross_val_score(clf, X[train_idx], y[train_idx], cv=cv)
    clf.fit(X[train_idx], y[train_idx])
    return clf, train_idx, np.sort(test_idx), scores


def train(
        subject: int,
        runs: list[int],
        config: PreprocConfig = DEFAULT,
        fit: FitConfig = FIT_DEFAULT,
        verbose: bool = True
) -> float:
    """
    Fit and persist a model for one subject and run group.

    Thin imperative shell over ``fit_pipeline``: loads the data, delegates
    the split and the fit, then writes the estimator together with
    everything ``predict`` needs to address the same epochs again. The held
    out partition is never scored here.

    Parameters
    ----------
    subject : int
        Subject number, 1-109.
    runs : list of int
        Runs sharing one T1/T2 semantics, as returned by ``runs_for``.
    config : PreprocConfig
        Preprocessing parameters; persisted so ``predict`` reproduces them.
    fit : FitConfig
        Split and estimator parameters. Not persisted.
    verbose : bool, default=True
        Print the per-fold scores and their mean in the subject's format.

    Returns
    -------
    cv_mean : float
        Mean validation accuracy on the training partition. Not a
        generalisation estimate — that is ``predict``'s job.

    Side Effects
    ------------
    Creates ``MODELS_DIR`` and writes ``model_path(subject, runs)``,
    overwriting any existing file without warning.
    """
    X, y, _ = load_dataset(subject, runs, config)
    clf, train_idx, test_idx, scores = fit_pipeline(X, y, fit)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": clf,
            "config": config,        # predict must preprocess identically
            "fit": fit,
            "subject": subject,
            "runs": runs,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "cv_scores": scores,
        },
        model_path(subject, runs),
    )

    if verbose:
        print(np.array2string(scores, precision=4, floatmode="fixed"))
        print(f"cross_val_score: {scores.mean():.4f}")

    return float(scores.mean())
