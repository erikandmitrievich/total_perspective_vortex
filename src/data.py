from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import mne


@dataclass(frozen=True)
class PreprocConfig:
    l_freq:     float = 7
    h_freq:     float = 30
    t_min:      float = -1.0
    t_max:      float = 4.0
    crop_t_min: float | None = 1.0
    crop_t_max: float | None = 2.0
    montage:    str = "standard_1005"
    fir_design: str = "firwin"


DEFAULT = PreprocConfig()


def load_raw(subject: int,
             runs: list[int],
             *,
             path: Path | None = None
             ) -> mne.io.BaseRaw:
    raise NotImplementedError


def preprocess(raw: mne.io.BaseRaw,
               config: PreprocConfig = DEFAULT
               ) -> mne.io.BaseRaw:
    raise NotImplementedError


def make_epochs(raw: mne.io.BaseRaw,
                event_id: dict[str, int],
                config: PreprocConfig = DEFAULT
                ) -> mne.Epochs:
    raise NotImplementedError
