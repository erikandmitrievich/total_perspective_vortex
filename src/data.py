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
    l_freq:     float = 7.0
    h_freq:     float = 30.0
    tmin:       float = -1.0
    tmax:       float = 4.0
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
    Standardise channel names, attach montage, band-pass filter.

    Operates on a copy. Deliberately applies no re-referencing: CAR imposes
    1ᵀZ = 0, making Sigma rank-63 and singular, which breaks the Cholesky
    step inside the generalised eigensolver.
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

    Parameters
    ----------
    """
    picks = pick_types(filtered.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")

    # Derive events explicitly. Epochs(raw, event_id=dict) does NOT map
    # descriptions -- it uses the default alphabetical codes and treats the
    # dict values as a code filter, silently selecting T0 and T1.
    events, _ = mne.events_from_annotations(filtered, event_id=EVENT_ID)

    epochs = Epochs(
        filtered,
        events=events,
        event_id=EVENT_ID,
        tmin=config.tmin, tmax=config.tmax,
        proj=True, picks=picks, baseline=None, preload=True,
    )

    dropped = [(i, d) for i, d in enumerate(epochs.drop_log)
               if d and d[0] != "IGNORED"]
    if dropped:
        raise RuntimeError(f"subject dropped {len(dropped)} epochs: {dropped}")

    return epochs


def _labels_from_events(
        epochs: mne.Epochs
) -> npt.NDArray[np.int_]:
    """
    Map pinned event codes to 0-based class labels.

    ``EVENT_ID`` fixes T1→1, T2→2 (code 0 is rejected by ``Epochs``), so
    the shift is by one and is not a magic offset.
    """
    return epochs.events[:, -1] - 1


def crop(
        epochs: mne.Epochs,
        config: PreprocConfig = DEFAULT
) -> mne.Epochs:
    """
    Extract the feature window. Returns a copy; ``epochs`` unmodified.
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
    """Load, preprocess, epoch and crop — the entry point every caller uses.

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Cropped epoch data.
    y : ndarray, shape (n_epochs,)
        Labels in {0, 1}.
    epochs : mne.Epochs
        Cropped epochs. Carries ``info`` for topomaps and ``events`` for
        the truth column in ``predict``.
    """
    raw = load_raw(subject, runs, path=path)
    epochs = crop(make_epochs(preprocess(raw, config), config), config)
    return epochs.get_data(copy=False), _labels_from_events(epochs), epochs
