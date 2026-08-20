import numpy as np

from src.experiments import EXPERIMENTS, SUBJECTS
from src.predict import score_stream
from src.train import fit_pipeline
from src.data import load_dataset, DEFAULT


def evaluate(subject, runs, config=DEFAULT, **kw):
    X, y, _ = load_dataset(subject, runs, config)
    clf, _, test_idx, scores = fit_pipeline(X, y, **kw)
    return score_stream(clf, X, y, test_idx, verbose=False)


def run_full_sweep():
    group_means = []

    for exp, (runs, label) in enumerate(EXPERIMENTS):
        accs = []
        for subject in SUBJECTS:
            try:
                acc = evaluate(subject, runs)
            except Exception as e:
                print(f"experiment {exp}: subject {subject:03d}: FAILED ({e!r})")
                continue
            accs.append(acc)
            print(f"experiment {exp}: subject {subject:03d}: "
                  f"accuracy = {acc:.4f}  ({label})")

        group_means.append(float(np.mean(accs)) if accs else float("nan"))

    print(f"\nMean accuracy of the six different experiments for "
          f"{len(SUBJECTS)} subjects:")
    for exp, (m, (_, label)) in enumerate(zip(group_means, EXPERIMENTS)):
        print(f"experiment {exp}:      accuracy = {m:.4f}  ({label})")
    print(f"\nMean accuracy of {len(EXPERIMENTS)} experiments: "
          f"{np.mean(group_means):.4f}")

    return group_means
