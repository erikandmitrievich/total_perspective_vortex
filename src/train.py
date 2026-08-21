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
    n_components: int = 4
    test_size:    float = 0.2
    n_splits:     int = 10
    seed:         int = 42


FIT_DEFAULT = FitConfig()

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

    Single definition of the pipeline: ``train``, ``predict`` and the sweep
    must all instantiate it here so that no caller can drift on
    hyperparameters.

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
    """TODO: write docstring
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
    """ TODO: rewrite
    Fit and persist a model for one subject/run set.

    Holds out a stratified test partition, cross-validates the whole
    pipeline on the remainder, refits on the remainder, and writes the
    fitted estimator plus everything ``predict`` needs to reproduce the
    split. The test partition is never touched here.

    Parameters
    ----------
    subject : int
        Subject number, 1-109.
    runs : list of int
        Runs sharing one T1/T2 semantics, as returned by ``runs_for``.
    config : PreprocConfig
        Preprocessing parameters; persisted so ``predict`` can reproduce them.
    verbose : bool, default=True
        Print the per-fold scores and their mean in the subject's format.

    Returns
    -------
    cv_mean : float
        Mean cross-validation accuracy on the training partition. Not a
        generalisation estimate — that is ``predict``'s job.

    Side
    ------------
    Creates ``MODELS_DIR`` and writes ``model_path(subject, runs)``,
    overwriting any existing file.
    """
    X, y, _ = load_dataset(subject, runs, config)
    clf, train_idx, test_idx, scores = fit_pipeline(X, y, fit)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": clf,
            "config": config,        # predict must preprocess identically
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
