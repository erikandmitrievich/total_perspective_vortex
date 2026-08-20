import time

import joblib
import numpy as np

from src.data import load_dataset
from src.train import model_path


def score_stream(clf, X, y, test_idx, *, verbose=True):
    """ TODO: write doc
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
            # print(f"epoch {k:02d}:     [{p}]       [{t}]    {p == t}")
        print(f"Accuracy: {acc:.4f}")
        print(f"max chunk latency: {max(latencies) * 1e3:.2f} ms")

    return acc


def predict(
        subject: int,
        runs: list[int],
        *,
        verbose: bool = True
) -> float:
    """ TODO: write doc
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
