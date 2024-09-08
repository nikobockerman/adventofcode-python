import importlib
import logging
import pathlib
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, assert_never

import joblib
import typer

from adventofcode.answers import ANSWERS

app = typer.Typer()
state = {"day_suffix": ""}


@app.callback()
def callback(
    verbosity: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            show_default=False,
            help="Increase log level. This option can be specified multiple "
            "times. Log levels by count of this flag: 0=WARNING, 1=INFO, 2=DEBUG.",
        ),
    ] = 0,
    day_suffix: Annotated[str, typer.Option("--day-suffix", "-s")] = "",
) -> None:
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbosity, logging.DEBUG
    )
    logging.basicConfig(level=level)
    state["day_suffix"] = day_suffix


@app.command(name="all")
def all_() -> None:
    sys.exit(_multiple_problems(ANSWERS.keys(), day_suffix=state["day_suffix"]))


@app.command(name="day")
def day_(day: int) -> None:
    sys.exit(_multiple_problems((day,), day_suffix=state["day_suffix"]))


class _Profiler(StrEnum):
    CProfile = "cProfile"
    Pyinstrument = "pyinstrument"


@app.command()
def single(
    day: int,
    problem: int,
    profiler: Annotated[_Profiler | None, typer.Option("-p", "--profiler")] = None,
) -> None:
    sys.exit(_specific_problem(day, state["day_suffix"], problem, profiler))


def _specific_problem(
    day: int, day_suffix: str, problem: int, profiler: _Profiler | None
) -> int:
    try:
        input_ = _get_problem_input(day, day_suffix, problem)
        if profiler is not None:
            output = _profiler_problem(input_, profiler)
        else:
            output = _exec_problem(input_)
        result = _process_output(output)
        if result.duration is not None:
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
        slowest = None
        for day in days:
            problems = ANSWERS.get(day, {})
            inputs = [
                _get_problem_input(day, day_suffix, problem) for problem in problems
            ]
            outputs = parallel(joblib.delayed(_exec_problem)(x) for x in inputs)
            results = (_process_output(x) for x in outputs)
            for result in results:
                passed = _report_one_of_many_problems(result)
                if all_passed is None:
                    all_passed = passed
                all_passed &= passed
                assert result.duration is not None
                if slowest is None:
                    slowest = result
                else:
                    assert slowest.duration is not None
                    if result.duration > slowest.duration:
                        slowest = result

    duration = time.perf_counter() - start

    if slowest is not None:
        print(
            f"Slowest: Day {slowest.day} Problem {slowest.problem}: "
            f"{slowest.duration:.3f}s"
        )

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
    duration: float | None
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
    duration: float | None
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
        output.day,
        output.problem,
        output.result,
        output.duration,
        str(answer) if answer is not None else None,
    )


def _exec_problem(input_: _ProblemInput) -> _ProblemOutput:
    start_time = time.perf_counter()
    result = input_.func(input_.input_str)
    duration = time.perf_counter() - start_time
    return _ProblemOutput(input_, duration, str(result))


def _profiler_problem(input_: _ProblemInput, profiler: _Profiler) -> _ProblemOutput:
    if profiler is _Profiler.CProfile:
        import cProfile
        import pstats

        with cProfile.Profile() as pr:
            result = input_.func(input_.input_str)
            stats = pstats.Stats(pr).strip_dirs()
            stats.sort_stats("tottime").print_stats(0.2)
            stats.sort_stats("cumtime").print_stats(0.2)
            return _ProblemOutput(input_, None, str(result))

    elif profiler is _Profiler.Pyinstrument:
        import pyinstrument
        import pyinstrument.util
        from pyinstrument.renderers.console import ConsoleRenderer

        with pyinstrument.profile(
            renderer=ConsoleRenderer(
                color=pyinstrument.util.file_supports_color(sys.stderr),
                unicode=pyinstrument.util.file_supports_unicode(sys.stderr),
                short_mode=True,
                show_all=True,
            )
        ):
            result = input_.func(input_.input_str)
            return _ProblemOutput(input_, None, str(result))
    else:
        assert_never(profiler)


if __name__ == "__main__":
    app()
