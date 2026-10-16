"""CLI: flipbudget correct <records.json> <audit.json> -- corrects a
benchmark's reported accuracy for scorer misclassification from the command
line, given a records file and an audit file in the schemas documented in
the README.
"""
from __future__ import annotations

import argparse
import json
import sys

from .inference import corrected_interval


def main() -> None:
    parser = argparse.ArgumentParser(prog="flipbudget")
    sub = parser.add_subparsers(dest="command", required=True)

    correct = sub.add_parser(
        "correct", help="correct one model's accuracy for scorer misclassification"
    )
    correct.add_argument("records", help="JSON file: a list of 0/1 scorer verdicts")
    correct.add_argument(
        "audit",
        help="JSON file: {'ystar': [0/1,...], 'yhat': [0/1,...]} paired audit labels",
    )
    correct.add_argument("--n-boot", type=int, default=1000)
    correct.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "correct":
        with open(args.records) as f:
            records = json.load(f)
        with open(args.audit) as f:
            audit = json.load(f)
        result = corrected_interval(
            records, audit["ystar"], audit["yhat"],
            n_boot=args.n_boot, seed=args.seed,
        )
        json.dump(result, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
