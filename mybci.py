import argparse
import sys

import mne

from src.experiments import runs_for
from src.predict import predict
from src.sweep import run_full_sweep
from src.train import train


def parse_args(
        argv=None
) -> argparse.Namespace:
    """
    Parse and validate the command-line interface.

    ``subject``, ``run`` and ``mode`` form an all-or-nothing group: all three
    select a single job, none selects the full sweep.

    Parameters
    ----------
    argv : list of str or None, default=None
        Argument list to parse. ``None`` defers to ``sys.argv[1:]``.

    Returns
    -------
    args : argparse.Namespace
        Holds ``subject``, ``run`` and ``mode`` — either all set or all None.

    Raises
    ------
    SystemExit
        On a partially filled group, ``subject`` outside 1-109, or an invalid
        ``mode``. ``run`` is not range-checked here.
    """
    parser = argparse.ArgumentParser(prog="mybci.py")
    parser.add_argument("subject", type=int, nargs="?", default=None,
                        help="Subject number (1-109)")
    parser.add_argument("run", type=int, nargs="?", default=None,
                        help="Experiment/run number")
    parser.add_argument("mode", choices=["train", "predict"], nargs="?", default=None,
                        help="train or predict")

    args = parser.parse_args(argv)

    given = [args.subject, args.run, args.mode]
    if any(v is not None for v in given) and any(v is None for v in given):
        parser.error("subject, run and mode must all be given together, or none at all")

    if args.subject is not None and not (1 <= args.subject <= 109):
        parser.error("subject must be between 1 and 109")

    return args


def main(
        argv=None
):
    """
    Entry point: dispatch to sweep, training or prediction.

    Runs the full sweep when no arguments are given, otherwise trains or
    predicts on the requested subject/run pair.

    Parameters
    ----------
    argv : list of str or None, default=None
        Argument list forwarded to ``parse_args``.
    """
    mne.set_log_level("ERROR")

    args = parse_args(argv)
    if args.subject is None:
        run_full_sweep()
    elif args.mode == "train":
        train(args.subject, runs_for(args.run))
    else:
        predict(args.subject, runs_for(args.run))


if __name__ == "__main__":
    main(sys.argv[1:])
