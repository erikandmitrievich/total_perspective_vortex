import numpy as np

from src.experiments import GROUPS, SUBJECTS
from src.predict import predict
from src.train import train


def run_full_sweep():
    group_means = []

    for exp, group in enumerate(GROUPS):
        accs = []
        for subject in SUBJECTS:
            train(subject, group, verbose=False)
            acc = predict(subject, group, verbose=False)
            accs.append(acc)
            print(f"experiment {exp}: subject {subject:03d}: accuracy = {acc:.4f}")

        group_means.append(float(np.mean(accs)))

    print("\nMean accuracy of the six different experiments for all "
          f"{len(SUBJECTS)} subjects:")
    for exp, m in enumerate(group_means):
        print(f"experiment {exp}:      accuracy = {m:.4f}")
    print(f"\nMean accuracy of {len(GROUPS)} experiments: {np.mean(group_means):.4f}")
