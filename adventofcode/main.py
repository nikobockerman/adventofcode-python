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
        days = list(ANSWERS.keys()) if args.day is None else [args.day]
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
    except SolverNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except IncorrectAnswerError as e:
        print(e, file=sys.stderr)
        return 2
    else:
        return 0


def one_of_many_problems(day: int, problem: int) -> bool:
    try:
        run_problem(day, problem, quiet=True)
    except IncorrectAnswerError as e:
        print(f"Day {day} Problem {problem}: FAIL: {e}")
        return False
    else:
        print(f"Day {day} Problem {problem}: PASS")
        return True


class SolverNotFoundError(RuntimeError):
    pass


class DayNotFoundError(SolverNotFoundError):
    def __init__(self, day: int) -> None:
        super().__init__(f"Solver for day {day} not found")


class ProblemNotFoundError(SolverNotFoundError):
    def __init__(self, problem: int) -> None:
        super().__init__(f"Solver for problem {problem} not found")


class IncorrectAnswerError(RuntimeError):
    def __init__(self, answer: Any, correct_answer: Any) -> None:
        super().__init__(f"Incorrect answer: {answer}. Correct is: {correct_answer}")


def run_problem(day: int, problem: int, *, quiet: bool = False) -> Any:
    try:
        mod = importlib.import_module(f"adventofcode.d{day}")
    except ModuleNotFoundError:
        raise DayNotFoundError(day) from None

    try:
        func = getattr(mod, f"p{problem}")
    except AttributeError:
        raise ProblemNotFoundError(problem) from None

    input_str = (
        (pathlib.Path(__file__).parent / f"input-d{day}.txt").read_text().strip()
    )

    result = func(input_str)

    def output(msg: str) -> None:
        if not quiet:
            print(msg)

    answer = ANSWERS.get(day, {}).get(problem)
    if answer is not None:
        if str(answer) != str(result):
            raise IncorrectAnswerError(result, answer)

        output(f"Answer is still correct: {result}")
    else:
        output(f"{result}")


if __name__ == "__main__":
    main()
