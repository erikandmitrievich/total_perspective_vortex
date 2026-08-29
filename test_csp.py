"""MyCSP against mne.decoding.CSP on one subject's epochs."""

import mne
import numpy as np
from mne.decoding import CSP

from src.csp import MyCSP
from src.data import load_dataset
from src.experiments import runs_for


N_COMPONENTS = 4
RTOL = 1e-6


def test_matches_mne_reference():
    """
    Compare MyCSP's log band-power features against MNE's on one subject.

    ``component_order="alternate"`` makes MNE select the same four filters
    — half from each end of the spectrum — but it orders them differently,
    so rows are sorted before comparison.
    """
    mne.set_log_level("ERROR")
    runs = runs_for(6)
    X, y, _ = load_dataset(1, runs)

    mine = MyCSP(n_components=N_COMPONENTS, log=True).fit(X, y)
    ref = CSP(n_components=N_COMPONENTS, log=True, reg=None,
              component_order="alternate").fit(X, y)

    A = np.sort(mine.transform(X), axis=1)
    B = np.sort(ref.transform(X), axis=1)
    dev = np.abs(A - B) / np.abs(B)

    print(f"subject 1, runs {runs}: {X.shape[0]} epochs, {X.shape[1]} channels,"
          f" class counts {np.bincount(y)}")
    print(f"max relative deviation: {dev.max():.2e}  (rtol {RTOL:.0e})")

    assert A.shape == (len(y), N_COMPONENTS)
    np.testing.assert_allclose(A, B, rtol=RTOL)


if __name__ == "__main__":
    test_matches_mne_reference()
    print("MyCSP matches mne.decoding.CSP")
