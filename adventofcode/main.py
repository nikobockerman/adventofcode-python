import importlib
import logging
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Optional

import typer
from typing_extensions import Annotated

from adventofcode.answers import ANSWERS

app = typer.Typer()


@app.command()
def main(
    day: Annotated[Optional[int], typer.Argument()] = None,
    problem: Annotated[Optional[int], typer.Argument()] = None,
    verbosity: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            show_default=False,
            help="Increase verbosity level. This option can be specified multiple "
            "times.",
        ),
    ] = 0,
    day_suffix: Annotated[str, typer.Option("--day-suffix", "-s")] = "",
) -> None:
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbosity, logging.DEBUG
    )
    logging.basicConfig(level=level)

    exit_code: int = 0
    if day is not None and problem is not None:
        exit_code = _specific_problem(day, day_suffix, problem)
    else:
        days = list(ANSWERS.keys()) if day is None else [day]
        all_passed = None
        start = time.perf_counter()
        for day in days:
            problems = ANSWERS.get(day, {})
            for problem in problems:
                passed = _one_of_many_problems(day, day_suffix, problem)
                if all_passed is None:
                    all_passed = passed
                all_passed &= passed
        duration = time.perf_counter() - start
        if all_passed is None:
            print("No answers known for requested day")
        elif all_passed:
            print(f"Finished with all passing. Duration {duration:.3f}s")
        else:
            print(f"Finished with failures. Duration {duration:.3f}s")
            exit_code = 1

    sys.exit(exit_code)


def _specific_problem(day: int, day_suffix: str, problem: int) -> int:
    try:
        result = _run_problem(day, day_suffix, problem)
        print(f"Duration: {result.duration:.3f}s")

        if result.incorrect:
            print(
                f"Incorrect answer: {result.answer}. "
                f"Correct is: {result.correct_answer}",
                file=sys.stderr,
            )
            return 2
        if result.correct:
            print(f"Answer is still correct: {result.answer}")
        else:
            print(result.answer)
    except _SolverNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    else:
        return 0


def _one_of_many_problems(day: int, day_suffix: str, problem: int) -> bool:
    result = _run_problem(day, day_suffix, problem)
    msg = f"Day {day:2} Problem {problem}: "
    msg += f"{result.duration:.3f}s: "
    if result.incorrect:
        msg += (
            f"FAIL: Incorrect answer: {result.answer}. "
            f"Correct is: {result.correct_answer}"
        )
    else:
        msg += "PASS"
    print(msg)
    return not result.incorrect


class _SolverNotFoundError(RuntimeError):
    pass


class _DayNotFoundError(_SolverNotFoundError):
    def __init__(self, day: int) -> None:
        super().__init__(f"Solver for day {day} not found")


class _ProblemNotFoundError(_SolverNotFoundError):
    def __init__(self, problem: int) -> None:
        super().__init__(f"Solver for problem {problem} not found")


@dataclass
class _ProblemResult:
    answer: str
    duration: float
    correct_answer: str | None

    @property
    def answer_known(self) -> bool:
        return self.correct_answer is not None

    @property
    def correct(self) -> bool:
        return self.answer_known and self.answer == self.correct_answer

    @property
    def incorrect(self) -> bool:
        return self.answer_known and self.answer != self.correct_answer


def _run_problem(day: int, day_suffix: str, problem: int) -> _ProblemResult:
    try:
        mod_name = f"adventofcode.d{day}{day_suffix}"
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        raise _DayNotFoundError(day) from None

    try:
        func = getattr(mod, f"p{problem}")
    except AttributeError:
        raise _ProblemNotFoundError(problem) from None

    input_str = (
        (pathlib.Path(__file__).parent / f"input-d{day}.txt").read_text().strip()
    )

    try:
        start_time = time.perf_counter()
        result = func(input_str)
        duration = time.perf_counter() - start_time
    except AssertionError as e:
        logging.critical(f"{mod_name}.p{problem}: Assertion failed: %s", e)
        raise

    answer = ANSWERS.get(day, {}).get(problem)
    return _ProblemResult(str(result), duration, str(answer))


if __name__ == "__main__":
    typer.run(main)
