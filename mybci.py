"""
Command-line entry point.

Validates arguments and dispatches to the sweep, ``train`` or ``predict``.
Holds no processing logic of its own.
"""

import argparse
import sys

import mne

from src.experiments import SUBJECTS, runs_for
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
        Holds ``subject``, ``run``, ``mode`` and ``runs`` — the first three
        either all set or all None, ``runs`` the base group ``run`` belongs
        to (``None`` when no job was selected)

    Raises
    ------
    SystemExit
        On a partially filled group, ``subject`` outside 1-109, ``run`` not
        a member of any ``BASE_GROUPS`` entry, or an invalid ``mode``. Run
        validity is membership, not range — the check is ``runs_for``'s and
        is surfaced here as a usage error.
    """
    parser = argparse.ArgumentParser(prog="mybci.py")
    parser.add_argument("subject", type=int, nargs="?", default=None,
                        help="Subject number (1-109)")
    parser.add_argument("run", type=int, nargs="?", default=None,
                        help="Experiment/run number")
    parser.add_argument("mode", choices=["train", "predict"], nargs="?",
                        default=None, help="train or predict")

    args = parser.parse_args(argv)

    given = [args.subject, args.run, args.mode]

    if any(v is None for v in given):
        if any(v is not None for v in given):
            parser.error(
                "subject, run and mode must all be given together, "
                "or none at all"
            )
        args.runs = None
        return args

    if args.subject not in SUBJECTS:
        parser.error("subject must be between "
                     f"{SUBJECTS.start} and {SUBJECTS.stop - 1}")
    try:
        args.runs = runs_for(args.run)
    except ValueError as e:
        parser.error(str(e))

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
        train(args.subject, args.runs)
    else:
        predict(args.subject, args.runs)


if __name__ == "__main__":
    main(sys.argv[1:])
