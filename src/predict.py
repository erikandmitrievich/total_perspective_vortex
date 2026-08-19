import time

import joblib
import numpy as np

from src.data import load_dataset
from src.train import model_path


def predict(
        subject: int,
        runs: list[int],
        *,
        verbose: bool = True
) -> float:
    path = model_path(subject, runs)
    if not path.exists():
        raise FileNotFoundError(f"no model at {path}; run `train` first")

    d = joblib.load(path)
    X, y, epochs = load_dataset(subject, runs, d["config"])

    n_expected = len(d["train_idx"]) + len(d["test_idx"])
    if len(y) != n_expected:
        raise RuntimeError(
            f"epoch count changed since training ({len(y)} vs {n_expected}); "
            f"test_idx no longer addresses the same epochs"
        )

    test_idx = np.sort(d["test_idx"])
    truth = epochs.events[test_idx, -1]
    clf = d["pipeline"]

    preds, latencies = [], []
    for i in test_idx:
        chunk = X[i:i + 1]                     # one epoch = one stream chunk
        t0 = time.perf_counter()
        p = int(clf.predict(chunk)[0])
        latencies.append(time.perf_counter() - t0)
        preds.append(p + 1)                    # {0,1} -> {1,2} for display

    preds = np.asarray(preds)
    acc = float((preds == truth).mean())

    if verbose:
        print("epoch nb: [prediction] [truth] equal?")
        for k, (p, t) in enumerate(zip(preds, truth)):
            print(f"epoch {k:02d}:     [{p}]       [{t}]    {p == t}")
        print(f"Accuracy: {acc:.4f}")
        print(f"max chunk latency: {max(latencies) * 1e3:.2f} ms")

    return acc
