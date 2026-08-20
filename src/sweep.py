import numpy as np

from src.experiments import GROUPS, SUBJECTS
from src.predict import predict
from src.train import train


def run_full_sweep():
    group_means = []

    for exp, (runs, label) in enumerate(GROUPS):
        accs = []
        for subject in SUBJECTS:
            try:
                train(subject, runs, verbose=False)
                acc = predict(subject, runs, verbose=False)
            except Exception as e:
                print(f"experiment {exp}: subject {subject:03d}: FAILED ({e!r})")
                continue
            accs.append(acc)
            print(f"experiment {exp}: subject {subject:03d}: "
                  f"accuracy = {acc:.4f}  ({label})")

        group_means.append(float(np.mean(accs)) if accs else float("nan"))

    print(f"\nMean accuracy of the six different experiments for "
          f"{len(SUBJECTS)} subjects:")
    for exp, (m, (_, label)) in enumerate(zip(group_means, GROUPS)):
        print(f"experiment {exp}:      accuracy = {m:.4f}  ({label})")
    print(f"\nMean accuracy of {len(GROUPS)} experiments: "
          f"{np.mean(group_means):.4f}")

    return group_means
