import argparse
import importlib
import logging
import pathlib
import sys
from typing import Any

from adventofcode.answers import ANSWERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("day", type=int, nargs="?")
    parser.add_argument("problem", type=int, nargs="?")
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    level = logging.INFO
    if args.debug:
        level = logging.DEBUG
    logging.basicConfig(level=level)

    exit_code: int = 0
    if args.day is not None and args.problem is not None:
        day: int = args.day
        problem: int = args.problem
        exit_code = specific_problem(day, problem)
    else:
        if args.day is None:
            days = list(ANSWERS.keys())
        else:
            days = [args.day]
        all_passed = None
        for day in days:
            problems = ANSWERS.get(day, {})
            for problem in problems:
                passed = one_of_many_problems(day, problem)
                if all_passed is None:
                    all_passed = passed
                all_passed &= passed
        if all_passed is None:
            print("No answers known for requested day")
        elif all_passed:
            print("Finished with all passing.")
        else:
            print("Finished with failures.")
            exit_code = 1

    sys.exit(exit_code)


def specific_problem(day: int, problem: int) -> int:
    try:
        run_problem(day, problem)
        return 0
    except ProblemNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except IncorrectAnswer as e:
        print(e, file=sys.stderr)
        return 2


def one_of_many_problems(day: int, problem: int) -> bool:
    try:
        run_problem(day, problem, quiet=True)
        print(f"Day {day} Problem {problem}: PASS")
        return True
    except IncorrectAnswer as e:
        print(f"Day {day} Problem {problem}: FAIL: {e}")
        return False


class ProblemNotFoundError(RuntimeError):
    pass


class IncorrectAnswer(RuntimeError):
    pass


def run_problem(day: int, problem: int, *, quiet: bool = False) -> Any:
    try:
        mod = importlib.import_module(f"adventofcode.d{day}")
    except ModuleNotFoundError:
        raise ProblemNotFoundError(f"Solver for day {day} not found")

    try:
        func = getattr(mod, f"p{problem}")
    except AttributeError:
        raise ProblemNotFoundError(f"Solver for problem {problem} not found")

    input = (pathlib.Path(__file__).parent / f"input-d{day}.txt").read_text().strip()

    result = func(input)

    def output(msg: str) -> None:
        if not quiet:
            print(msg)

    answer = ANSWERS.get(day, {}).get(problem)
    if answer is not None:
        if str(answer) != str(result):
            raise IncorrectAnswer(f"Incorrect answer: {result}. Correct is: {answer}")
        else:
            output(f"Answer is still correct: {result}")
    else:
        output(f"{result}")


if __name__ == "__main__":
    main()
