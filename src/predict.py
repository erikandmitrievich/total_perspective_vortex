import time

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.pipeline import Pipeline

from src.data import load_dataset
from src.train import model_path


def score_stream(
        clf: Pipeline,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.int_],
        test_idx: npt.NDArray[np.intp],
        *,
        verbose=True
) -> float:
    """
    Replay held-out epochs one at a time and score the predictions.

    The loop is the "stream": each epoch is handed to ``clf.predict``
    alone, so no held-out epoch can influence the classification of
    another and the per-chunk cost is measurable. Indices are consumed in
    the order given — sorted by ``fit_pipeline``, i.e. recording order.

    Parameters
    ----------
    clf : Pipeline
        Fitted CSP to LDA estimator.
    X : ndarray, shape (n_epochs, n_channels, n_times)
        All epochs of the run group; only ``test_idx`` rows are read.
    y : ndarray of int, shape (n_epochs,)
        Labels in {0, 1}, row-aligned with ``X``.
    test_idx : ndarray of intp
        Held-out rows to replay.
    verbose : bool, default=True
        Print the per-epoch table and the accuracy in the subject's
        format. Predictions and truth are printed shifted back to the
        1/2 convention of the annotations.

    Returns
    -------
    acc : float
        Fraction of held-out epochs classified correctly.
    """
    preds, latencies = [], []
    for i in test_idx:
        t0 = time.perf_counter()
        preds.append(int(clf.predict(X[i:i+1])[0]))
        latencies.append(time.perf_counter() - t0)
    preds = np.asarray(preds)
    truth = y[test_idx]
    acc = float((preds == truth).mean())

    if verbose:
        print("epoch nb: [prediction] [truth] equal?")
        for k, (p, t) in enumerate(zip(preds, truth)):
            print(f"epoch {k:02d}:     [{p+1}]       [{t+1}]    {p == t}")
        print(f"Accuracy: {acc:.4f}")
        print(f"max chunk latency: {max(latencies) * 1e3:.2f} ms")

    return acc


def predict(
        subject: int,
        runs: list[int],
        *,
        verbose: bool = True
) -> float:
    """
    Score a persisted model on its own held-out partition.

    Reloads the data under the *persisted* config rather than the current
    default, so a change to ``PreprocConfig`` cannot silently re-score an
    old model on differently preprocessed epochs.

    Parameters
    ----------
    subject : int
        Subject number, 1-109.
    runs : list of int
        Runs the model was trained on; selects the model file.
    verbose : bool, default=True
        Forwarded to ``score_stream``.

    Returns
    -------
    acc : float
        Held-out accuracy.

    Raises
    ------
    FileNotFoundError
        If no model exists for this subject and run group.
    RuntimeError
        If the epoch count differs from training. ``test_idx`` is a set of
        positions, not identities, so a changed count means the stored
        indices no longer address the epochs they were chosen from —
        scoring would proceed and return a meaningless number.
    """
    path = model_path(subject, runs)
    if not path.exists():
        raise FileNotFoundError(f"no model at {path}; run `train` first")

    d = joblib.load(path)
    X, y, _ = load_dataset(subject, runs, d["config"])

    n_expected = len(d["train_idx"]) + len(d["test_idx"])
    if len(y) != n_expected:
        raise RuntimeError(
            f"epoch count changed since training ({len(y)} vs {n_expected}); "
            f"test_idx no longer addresses the same epochs"
        )

    return score_stream(d["pipeline"], X, y, d["test_idx"], verbose=verbose)
