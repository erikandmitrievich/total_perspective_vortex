import sys
import argparse

from src.train import train
from src.predict import predict
from src.sweep import run_full_sweep


def parse_args(argv=None) -> argparse.Namespace:
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


def main(argv=None):
    args = parse_args(argv)
    if args.subject is None:
        run_full_sweep()
    elif args.mode == "train":
        train(args.subject, args.run)
    else:
        predict(args.subject, args.run)


if __name__ == "__main__":
    main(sys.argv[1:])
