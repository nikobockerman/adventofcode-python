import importlib
import logging
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import joblib
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
        days = (ANSWERS.keys()) if day is None else [day]
        exit_code = _multiple_problems(days, day_suffix=day_suffix)

    sys.exit(exit_code)


def _specific_problem(day: int, day_suffix: str, problem: int) -> int:
    try:
        result = _process_output(
            _exec_problem(_get_problem_input(day, day_suffix, problem))
        )
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


def _multiple_problems(days: Iterable[int], day_suffix: str) -> int:
    all_passed = None
    start = time.perf_counter()
    with joblib.Parallel(n_jobs=-1) as parallel:
        for day in days:
            problems = ANSWERS.get(day, {})
            inputs = list(
                _get_problem_input(day, day_suffix, problem) for problem in problems
            )
            outputs = parallel(joblib.delayed(_exec_problem)(x) for x in inputs)
            results = map(lambda x: _process_output(x), outputs)
            for result in results:
                passed = _report_one_of_many_problems(result)
                if all_passed is None:
                    all_passed = passed
                all_passed &= passed
    duration = time.perf_counter() - start

    if all_passed is None:
        print(f"No answers known for requested day. Duration {duration:.3f}s")
        return 0

    if all_passed:
        print(f"Finished with all passing. Duration {duration:.3f}s")
        return 0

    print(f"Finished with failures. Duration {duration:.3f}s")
    return 1


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
    day: int
    problem: int

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


def _report_one_of_many_problems(result: _ProblemResult) -> bool:
    msg = f"Day {result.day:2} Problem {result.problem}: "
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


@dataclass(frozen=True, slots=True)
class _ProblemInput:
    day: int
    problem: int
    func: Callable[[str], Any]
    input_str: str


@dataclass(frozen=True, slots=True)
class _ProblemOutput:
    input_: _ProblemInput
    duration: float
    result: str

    @property
    def day(self) -> int:
        return self.input_.day

    @property
    def problem(self) -> int:
        return self.input_.problem


def _get_problem_input(day: int, day_suffix: str, problem: int) -> _ProblemInput:
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

    return _ProblemInput(day, problem, func, input_str)


def _process_output(output: _ProblemOutput) -> _ProblemResult:
    answer = ANSWERS.get(output.day, {}).get(output.problem)
    return _ProblemResult(
        output.day, output.problem, output.result, output.duration, str(answer)
    )


def _exec_problem(input_: _ProblemInput) -> _ProblemOutput:
    start_time = time.perf_counter()
    result = input_.func(input_.input_str)
    duration = time.perf_counter() - start_time
    return _ProblemOutput(input_, duration, str(result))


if __name__ == "__main__":
    typer.run(main)
