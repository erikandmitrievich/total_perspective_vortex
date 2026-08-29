"""
Before/after PSD figure for the preprocessing step.

Kept out of ``mybci.py``: the plot is a deliverable in its own right and
pulls in matplotlib, which the classification path never needs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne

from src.data import DEFAULT, PreprocConfig, load_raw, preprocess
from src.experiments import SUBJECTS, runs_for


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figs"


def parse_args(
        argv=None
) -> argparse.Namespace:
    """
    Parse and validate the command-line interface.

    Both arguments are optional; the defaults select a subject and run that
    exist for every installation, so a bare invocation produces a figure.

    Parameters
    ----------
    argv : list of str or None, default=None
        Argument list to parse. ``None`` defers to ``sys.argv[1:]``.

    Returns
    -------
    args : argparse.Namespace
        Holds ``subject``, ``run`` and ``runs`` — the last being the base
        group ``run`` belongs to.

    Raises
    ------
    SystemExit
        On ``subject`` outside 1-109 or a ``run`` that is not one of the
        twelve motor-imagery runs.
    """
    parser = argparse.ArgumentParser(
        prog="visualise.py",
        description="Plot the power spectral density before and after "
                    "preprocessing, on one shared scale.",
    )
    parser.add_argument("subject", type=int, nargs="?", default=1,
                        help="Subject number (1-109)")
    parser.add_argument("run", type=int, nargs="?", default=6,
                        help="Motor-imagery run number (3-14)")

    args = parser.parse_args(argv)

    if args.subject not in SUBJECTS:
        parser.error(f"subject must be between {SUBJECTS.start} "
                     f"and {SUBJECTS.stop - 1}")

    try:
        args.runs = runs_for(args.run)
    except ValueError as e:
        parser.error(str(e))

    return args


def plot_psd_comparison(
        raw: mne.io.BaseRaw,
        filtered: mne.io.BaseRaw,
        config: PreprocConfig = DEFAULT
) -> plt.Figure:
    """
    Stack the before/after spectra on a single shared decibel axis.

    Two independently autoscaled figures would each rescale to their own
    content and hide the depth of the stopband; the y-limits are therefore
    unioned across both panels after plotting. ``show=False`` keeps the
    figure alive for ``savefig`` — the MNE default opens a blocking window
    on an interactive backend and can tear the figure down before it is
    written.

    ``fmax`` is clipped to the Nyquist frequency: four subjects are recorded
    at 128 Hz rather than 160 Hz, and a fixed 80 Hz upper bound would exceed
    their spectrum.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Unprocessed recording from ``load_raw``.
    filtered : mne.io.BaseRaw
        Output of ``preprocess`` on the same recording.
    config : PreprocConfig
        Supplies the band edges drawn as vertical markers.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Two stacked axes, raw above filtered. The caller owns it and is
        responsible for closing it.
    """
    fmax = min(80.0, raw.info["sfreq"] / 2.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    panels = (
        (axes[0], raw, "Raw"),
        (axes[1], filtered,
         f"Band-pass {config.l_freq:g}-{config.h_freq:g} Hz"),
    )

    for ax, data, title in panels:
        data.compute_psd(fmax=fmax).plot(axes=[ax], show=False)
        ax.set_title(title, loc="left")
        for edge in (config.l_freq, config.h_freq):
            ax.axvline(edge, color="k", linestyle="--", linewidth=1, alpha=0.6)

    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)

    fig.suptitle("Power spectral density, before and after preprocessing")
    return fig


def main(
        argv=None
) -> Path:
    """
    Entry point: load one run group, preprocess it, write the figure.

    Parameters
    ----------
    argv : list of str or None, default=None
        Argument list forwarded to ``parse_args``.

    Returns
    -------
    path : Path
        The PNG written.

    Side Effects
    ------------
    Creates ``FIG_DIR`` and writes ``S{subject:03d}_R{run}_psd.png``,
    overwriting any existing file without warning.
    """
    mne.set_log_level("ERROR")

    args = parse_args(argv)

    raw = load_raw(args.subject, args.runs)
    filtered = preprocess(raw, DEFAULT)

    fig = plot_psd_comparison(raw, filtered, DEFAULT)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"S{args.subject:03d}_R{args.run}_psd.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
