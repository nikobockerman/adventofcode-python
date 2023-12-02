import logging
import pathlib
from typing import Iterable

STR_DIGITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def find_first_and_last_int(input: str, use_texts: bool) -> tuple[str, str]:
    first: str | None = None
    last: str | None = None
    for ind, c in enumerate(input):
        val = None
        if c.isdigit():
            val = c
        elif use_texts:
            for digit_ind, str_digit in enumerate(STR_DIGITS):
                if input.find(str_digit, ind, ind + len(str_digit)) >= 0:
                    val = str(digit_ind + 1)
                    break

        if val is not None:
            if first is None:
                first = val
                last = val
            else:
                last = val

    assert first is not None and last is not None
    return first, last


def int_chars_to_int(s1: str, s2: str) -> int:
    return int(s1 + s2)


def p1_ints(lines: Iterable[str], use_texts: bool) -> Iterable[int]:
    for line in lines:
        logging.debug(f"{line=}")
        value = int_chars_to_int(*find_first_and_last_int(line, use_texts))
        logging.debug(f"{value=}")
        yield value


def p1() -> None:
    lines = (pathlib.Path(__file__).parent / "input1-raw.txt").read_text().splitlines()
    print(sum(p1_ints(lines, False)))


def p2() -> None:
    lines = (pathlib.Path(__file__).parent / "input1-raw.txt").read_text().splitlines()
    print(sum(p1_ints(lines, True)))
