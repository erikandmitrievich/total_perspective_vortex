"""
Loading, preprocessing, epoching — everything upstream of the pipeline.

``load_dataset`` is the single path from subject/runs to arrays; the stages
it composes are exported only because visualisation needs the intermediate
``raw`` and ``filtered`` objects.
"""

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import numpy.typing as npt
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


@dataclass(frozen=True)
class PreprocConfig:
    """
    Preprocessing parameters shared by training and prediction.

    Frozen so a config can be persisted alongside a fitted model and
    compared by value: ``predict`` must reproduce the exact pipeline the
    model was trained under.

    Attributes
    ----------
    l_freq, h_freq : float
        Band-pass edges in Hz. Defaults cover mu and beta (7-30 Hz), the
        bands carrying motor-imagery power modulation.
    tmin, tmax : float
        Feature window in seconds relative to event onset. ``tmax=2.0``
        keeps the window short enough to fit inside every trial; a longer
        window makes epochs adjacent to a run boundary overrun it, and MNE
        drops them without raising.
    montage : str
        Standard montage name passed to ``make_standard_montage``.
    fir_design : str
        FIR design forwarded to ``Raw.filter``.
    """
    l_freq:     float = 7.0
    h_freq:     float = 30.0
    tmin:       float = 1.0
    tmax:       float = 2.0
    montage:    str = "standard_1005"
    fir_design: str = "firwin"


DEFAULT = PreprocConfig()
"""
Module-level default configuration.

Every loader takes ``config=DEFAULT``, so an unqualified call and an
explicitly configured call cannot diverge on anything but what the caller
passed.
"""

EVENT_ID = {"T1": 1, "T2": 2}
"""
Annotation description to event code, pinned.

T0 (rest) is deliberately absent: only the two task classes are epoched.
These codes are what ``_labels_from_events`` shifts to {0, 1}, so the
mapping and the label convention are one decision, not two.
"""


def load_raw(
        subject: int,
        runs: list[int],
        *,
        path: Path | None = None
) -> mne.io.BaseRaw:
    """
    Load and concatenate the requested runs for one subject.

    Performs no filtering or referencing — returns raw data suitable for
    a "before" PSD plot.

    Parameters
    ----------
    subject : int
        Subject number, 1-109.
    runs : list of int
        Run numbers to concatenate.
    path : Path or None
        Download/cache root forwarded to ``eegbci.load_data``.

    Returns
    -------
    raw : mne.io.BaseRaw
        Concatenated, preloaded recording.
    """
    raw_fnames = eegbci.load_data(subject, runs, path=path)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    return raw


def preprocess(
        raw: mne.io.BaseRaw,
        config: PreprocConfig = DEFAULT
) -> mne.io.BaseRaw:
    """
    Standardise channel names, attach a montage, band-pass filter.

    Deliberately applies no re-referencing. CAR imposes 1ᵀZ = 0, making
    Sigma rank-deficient and singular, which breaks the Cholesky step
    inside the generalised eigensolver. The band-pass is also what
    licenses the zero-mean assumption in ``MyCSP._covariance``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Concatenated recording from ``load_raw``.
    config : PreprocConfig
        Supplies the filter edges, montage and FIR design.

    Returns
    -------
    filtered : mne.io.BaseRaw
        Preprocessed copy; ``raw`` is left unmodified.
    """
    filtered = raw.copy()
    eegbci.standardize(filtered)
    filtered.set_montage(make_standard_montage(config.montage))
    filtered.filter(config.l_freq, config.h_freq,
                    fir_design=config.fir_design,
                    skip_by_annotation="edge")
    return filtered


def make_epochs(
        filtered: mne.io.BaseRaw,
        config: PreprocConfig = DEFAULT
) -> mne.Epochs:
    """
    Slice preprocessed data into labelled epochs.

    Events are derived explicitly via ``events_from_annotations`` with a
    pinned ``EVENT_ID``. Passing a dict straight to ``Epochs(event_id=...)``
    does not map annotation descriptions — it filters on the default
    alphabetical codes, which silently selects T0 and T1 instead.

    Parameters
    ----------
    filtered : mne.io.BaseRaw
        Output of ``preprocess``.
    config : PreprocConfig
        Supplies the epoch bounds ``tmin``/``tmax``.

    Returns
    -------
    epochs : mne.Epochs
        Preloaded epochs over EEG channels only, no baseline correction.
    """
    picks = pick_types(filtered.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")

    events, _ = mne.events_from_annotations(filtered, event_id=EVENT_ID)

    epochs = Epochs(
        filtered,
        events=events,
        event_id=EVENT_ID,
        tmin=config.tmin, tmax=config.tmax,
        proj=True, picks=picks, baseline=None, preload=True,
    )

    return epochs


def _labels_from_events(
        epochs: mne.Epochs
) -> npt.NDArray[np.int_]:
    """
    Map pinned event codes to 0-based class labels.

    ``EVENT_ID`` fixes T1→1 and T2→2, so the shift is a consequence of the
    pinned mapping rather than a magic offset.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs whose ``events`` column was built from ``EVENT_ID``.

    Returns
    -------
    y : ndarray of int, shape (n_epochs,)
        Labels in {0, 1}, row-aligned with ``epochs``.
    """
    return epochs.events[:, -1] - 1


def load_dataset(
        subject: int,
        runs: list[int],
        config: PreprocConfig = DEFAULT,
        *,
        path: Path | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], mne.Epochs]:
    """
    Load, preprocess and epoch — the entry point every caller uses.

    Single definition of the preprocessing path. Tests and pipeline must
    both come through here; divergence between them is what previously
    let the oracle certify CSP on data the pipeline never saw.

    Parameters
    ----------
    subject : int
        Subject number, 1-109.
    runs : list of int
        Runs sharing one T1/T2 semantics, as returned by ``runs_for``.
    config : PreprocConfig
        Preprocessing parameters applied end to end.
    path : Path or None
        Download/cache root forwarded to ``eegbci.load_data``.

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Epoch data. This is the epochs' internal buffer, not a
        copy: mutating ``X`` mutates ``epochs``.
    y : ndarray of int, shape (n_epochs,)
        Labels in {0, 1}.
    epochs : mne.Epochs
        Carries ``info`` for topomaps and ``events`` for the truth
        column in ``predict``.
    """
    raw = load_raw(subject, runs, path=path)
    epochs = make_epochs(preprocess(raw, config), config)
    return epochs.get_data(copy=False), _labels_from_events(epochs), epochs
