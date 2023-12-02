import argparse
import importlib
import logging
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("day", type=int, default=1)
    parser.add_argument("problem", type=int, default=1)
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    level = logging.INFO
    if args.debug:
        level = logging.DEBUG
    logging.basicConfig(level=level)

    day: int = args.day
    problem: int = args.problem

    try:
        mod = importlib.import_module(f"adventofcode.d{day}")
    except ModuleNotFoundError:
        print(f"Solver for day {day} not found")
        sys.exit(1)

    try:
        func = getattr(mod, f"p{problem}")
    except AttributeError:
        print(f"Solver for problem {problem} not found")
        sys.exit(1)

    input = (pathlib.Path(__file__).parent / f"input-d{day}.txt").read_text()

    result = func(input)
    print(result)


if __name__ == "__main__":
    main()
