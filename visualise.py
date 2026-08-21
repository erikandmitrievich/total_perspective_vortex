import argparse

import mne

from pathlib import Path
from src.data import load_raw, preprocess
from src.experiments import runs_for

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    mne.set_log_level("ERROR")
    p = argparse.ArgumentParser(prog="visualize.py")
    p.add_argument("subject", type=int, nargs="?", default=1)
    p.add_argument("run", type=int, nargs="?", default=6)
    args = p.parse_args(argv)
    raw = load_raw(args.subject, runs_for(args.run))
    filtered = preprocess(raw)
    fig_raw = raw.compute_psd(fmax=80).plot()
    fig_filtered = filtered.compute_psd(fmax=80).plot()
    fig_raw.savefig(FIG_DIR/f"S{args.subject:03d}_R{args.run}_raw.png")
    fig_filtered.savefig(FIG_DIR/f"S{args.subject:03d}_R{args.run}_filtered.png")


if __name__ == "__main__":
    main()
