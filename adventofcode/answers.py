from collections.abc import Iterator
from typing import Literal, NewType, overload

from attrs import frozen

type Problem = Literal[1, 2]
type AnswerType = int

_ANSWERS: dict[int, dict[int, dict[Problem, AnswerType]]] = {
    2023: {
        1: {1: 54239, 2: 55343},
        2: {1: 1931, 2: 83105},
        3: {1: 556367, 2: 89471771},
        4: {1: 20107, 2: 8172507},
        5: {1: 111627841, 2: 69323688},
        6: {1: 1108800, 2: 36919753},
        7: {1: 251106089, 2: 249620106},
        8: {1: 16271, 2: 14265111103729},
        9: {1: 1731106378, 2: 1087},
        10: {1: 6951, 2: 563},
        11: {1: 9742154, 2: 411142919886},
        12: {1: 7173, 2: 29826669191291},
        13: {1: 37561, 2: 31108},
        14: {1: 102497, 2: 105008},
        15: {1: 515495, 2: 229349},
        16: {1: 7482, 2: 7896},
        17: {1: 870, 2: 1063},
        18: {1: 45159, 2: 134549294799713},
        19: {1: 406849, 2: 138625360533574},
        20: {1: 731517480, 2: 244178746156661},
    },
}

Year = NewType("Year", int)
Day = NewType("Day", int)


@frozen
class ProblemId:
    year: Year
    day: Day
    problem: Problem


@frozen
class Answer(ProblemId):
    answer: AnswerType


ANSWERS: dict[Year, dict[Day, dict[Problem, Answer]]] = {
    (y := Year(year)): {
        (d := Day(day)): {
            problem: Answer(year=y, day=d, problem=problem, answer=answer)
            for problem, answer in problems.items()
        }
        for day, problems in days.items()
    }
    for year, days in _ANSWERS.items()
}


def get_from_id(id_: ProblemId, /) -> Answer | None:
    return ANSWERS.get(id_.year, {}).get(id_.day, {}).get(id_.problem)


@overload
def get() -> Iterator[Answer]: ...
@overload
def get(year: Year, /) -> Iterator[Answer]: ...
@overload
def get(year: Year, day: Day, /) -> Iterator[Answer]: ...


def get(year: Year | None = None, day: Day | None = None) -> Iterator[Answer]:
    if year is None:
        for year_ in ANSWERS:
            yield from get(year_)
        return

    if day is None:
        days = ANSWERS[year]
        for day_ in days:
            yield from get(year, day_)
        return

    yield from ANSWERS[year][day].values()
