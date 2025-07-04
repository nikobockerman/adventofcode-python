from __future__ import annotations

import importlib
import logging
import pathlib
import sys
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, TypeIs, assert_never

import joblib
import typer
from attrs import define, frozen

from adventofcode import answers

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

app = typer.Typer(no_args_is_help=True)
state = {"day_suffix": ""}

YEAR = answers.Year(2023)


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
    sys.exit(_multiple_problems(answers.get(), day_suffix=state["day_suffix"]))


@app.command(name="day", no_args_is_help=True)
def day_(day: int) -> None:
    sys.exit(
        _multiple_problems(
            answers.get(YEAR, answers.Day(day)), day_suffix=state["day_suffix"]
        )
    )


class _Profiler(StrEnum):
    CProfile = "cProfile"
    Pyinstrument = "pyinstrument"


class _Problem(StrEnum):
    _1 = "1"
    _2 = "2"


@app.command(no_args_is_help=True)
def single(
    day: int,
    problem: _Problem,
    profiler: Annotated[_Profiler | None, typer.Option("-p", "--profiler")] = None,
) -> None:
    problem_: answers.Problem = problem  # type: ignore[reportAssignmentType, assignment]
    sys.exit(
        _specific_problem(
            answers.ProblemId(YEAR, answers.Day(day), problem_),
            state["day_suffix"],
            profiler,
        )
    )


def _specific_problem(
    id_: answers.ProblemId,
    day_suffix: str,
    profiler: _Profiler | None,
) -> int:
    try:
        input_ = _get_problem_input(id_, day_suffix)
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


def _multiple_problems(answers: Iterable[answers.Answer], day_suffix: str) -> int:
    all_passed = None
    slowest = None
    start = time.perf_counter()
    with joblib.Parallel(n_jobs=-1, return_as="generator") as parallel:
        inputs = [_get_problem_input(answer, day_suffix) for answer in answers]
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
            f"Slowest: {slowest.id.year} {slowest.id.day:2} {slowest.id.problem}: "
            f"{slowest.duration:.3f}s"
        )

    if all_passed is None:
        print(f"No answers known. Duration {duration:.3f}s")
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


@define
class _ProblemResult:
    id: answers.ProblemId
    answer: answers.AnswerType
    duration: float | None
    correct_answer: answers.AnswerType | None

    @property
    def correct(self) -> bool:
        return self.correct_answer is not None and self.correct_answer == self.answer

    @property
    def incorrect(self) -> bool:
        return self.correct_answer is not None and self.answer != self.correct_answer


def _report_one_of_many_problems(result: _ProblemResult) -> bool:
    msg = f"{result.id.year} {result.id.day:2} {result.id.problem}: "
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


@frozen
class _ProblemInput:
    id: answers.ProblemId
    func: Callable[[str], Any]
    input_str: str


@frozen
class _ProblemOutput:
    id_: answers.ProblemId
    duration: float | None
    result: answers.AnswerType


def _get_problem_input(id_: answers.ProblemId, day_suffix: str) -> _ProblemInput:
    try:
        mod_name = f"adventofcode.y{id_.year}.d{id_.day}{day_suffix}"
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        raise _DayNotFoundError(id_.day) from None

    try:
        func = getattr(mod, f"p{id_.problem}")
    except AttributeError:
        raise _ProblemNotFoundError(id_.problem) from None

    input_str = (
        (pathlib.Path(__file__).parent / f"y{id_.year}" / f"input-d{id_.day}.txt")
        .read_text()
        .strip()
    )

    return _ProblemInput(id_, func, input_str)


def _process_output(output: _ProblemOutput) -> _ProblemResult:
    answer = answers.get_from_id(output.id_)
    return _ProblemResult(
        output.id_,
        output.result,
        output.duration,
        answer.answer if answer is not None else None,
    )


class InvalidResultTypeError(TypeError):
    def __init__(self, result_type: type) -> None:
        super().__init__(f"Invalid result type: {result_type}")


def is_valid_result_type(result: Any) -> TypeIs[answers.AnswerType]:  # noqa: ANN401
    return isinstance(result, int)


def _exec_problem(input_: _ProblemInput) -> _ProblemOutput:
    start_time = time.perf_counter()
    result: Any = input_.func(input_.input_str)
    duration = time.perf_counter() - start_time
    if not is_valid_result_type(result):
        raise InvalidResultTypeError(type(result))  # type: ignore[reportUnknownArgumentType]

    return _ProblemOutput(input_.id, duration, result)


def _profiler_problem(input_: _ProblemInput, profiler: _Profiler) -> _ProblemOutput:
    if profiler is _Profiler.CProfile:
        import cProfile  # noqa: PLC0415
        import pstats  # noqa: PLC0415

        with cProfile.Profile() as pr:
            result = input_.func(input_.input_str)
            stats = pstats.Stats(pr).strip_dirs()
            stats.sort_stats("tottime").print_stats(0.2)
            stats.sort_stats("cumtime").print_stats(0.2)
            if not is_valid_result_type(result):
                raise InvalidResultTypeError(type(result))  # type: ignore[reportUnknownArgumentType]
            return _ProblemOutput(input_.id, None, result)

    elif profiler is _Profiler.Pyinstrument:
        import pyinstrument  # noqa: PLC0415
        import pyinstrument.util  # noqa: PLC0415
        from pyinstrument.renderers.console import ConsoleRenderer  # noqa: PLC0415

        with pyinstrument.profile(
            renderer=ConsoleRenderer(
                color=pyinstrument.util.file_supports_color(sys.stderr),
                unicode=pyinstrument.util.file_supports_unicode(sys.stderr),
                short_mode=True,
                show_all=True,
            )
        ):
            result = input_.func(input_.input_str)
            if not is_valid_result_type(result):
                raise InvalidResultTypeError(type(result))  # type: ignore[reportUnknownArgumentType]
            return _ProblemOutput(input_.id, None, result)

    assert_never(profiler)


if __name__ == "__main__":
    app()
