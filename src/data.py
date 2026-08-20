from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import mne


from mne import Epochs, pick_types
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage


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
        Epoch bounds in seconds relative to event onset.
    crop_tmin, crop_tmax : float or None
        Feature window inside the epoch. ``crop_tmin=None`` disables cropping.
    montage : str
        Standard montage name passed to ``make_standard_montage``.
    fir_design : str
        FIR design forwarded to ``Raw.filter``.
    """
    l_freq:     float = 7.0
    h_freq:     float = 30.0
    tmin:       float = -1.0
    tmax:       float = 2.0
    crop_tmin:  float | None = 1.0
    crop_tmax:  float | None = 2.0
    montage:    str = "standard_1005"
    fir_design: str = "firwin"


DEFAULT = PreprocConfig()
EVENT_ID = {"T1": 1, "T2": 2}


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

    Raises
    ------
    RuntimeError
        If any epoch was dropped for a reason other than ``IGNORED``.
        Dropped epochs would desynchronise ``X`` from any externally held
        index arrays, so this fails loudly rather than shrinking the set.
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


def crop(
        epochs: mne.Epochs,
        config: PreprocConfig = DEFAULT
) -> mne.Epochs:
    """
    Extract the feature window.

    Parameters
    ----------
    epochs : mne.Epochs
        Full-length epochs from ``make_epochs``.
    config : PreprocConfig
        Supplies ``crop_tmin``/``crop_tmax``.

    Returns
    -------
    cropped : mne.Epochs
        A cropped copy, leaving ``epochs`` unmodified — except when
        ``crop_tmin is None``, where ``epochs`` itself is returned.
    """
    if config.crop_tmin is None:
        return epochs
    return epochs.copy().crop(config.crop_tmin, config.crop_tmax)


def load_dataset(
        subject: int,
        runs: list[int],
        config: PreprocConfig = DEFAULT,
        *,
        path: Path | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], mne.Epochs]:
    """
    Load, preprocess, epoch and crop — the entry point every caller uses.

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
        Cropped epoch data. This is the epochs' internal buffer, not a
        copy: mutating ``X`` mutates ``epochs``.
    y : ndarray of int, shape (n_epochs,)
        Labels in {0, 1}.
    epochs : mne.Epochs
        Cropped epochs. Carries ``info`` for topomaps and ``events`` for
        the truth column in ``predict``.
    """
    raw = load_raw(subject, runs, path=path)
    epochs = crop(make_epochs(preprocess(raw, config), config), config)
    return epochs.get_data(copy=False), _labels_from_events(epochs), epochs
