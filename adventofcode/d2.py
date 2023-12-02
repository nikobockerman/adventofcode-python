import argparse
import logging
import pathlib
from typing import Iterable


def parse_input(lines: Iterable[str]) -> Iterable[tuple[int, list[dict[str, int]]]]:
    for line in lines:
        g_id, rounds = line[5:].split(":")
        logging.debug(f"{line=}")

        yield int(g_id), [
            {
                color_count[1]: int(color_count[0])
                for color_count in (
                    color_count_str.strip().split(" ")
                    for color_count_str in round.strip().split(",")
                )
            }
            for round in rounds.strip().split(";")
        ]


def p1(input: str):
    d = parse_input(input.splitlines())

    def maxes():
        for g_id, rounds in d:
            max_counts: dict[str, int] = {}
            for round in rounds:
                for color, count in round.items():
                    if color not in max_counts or max_counts[color] < count:
                        max_counts[color] = count
            yield g_id, max_counts

    return sum(
        g_id
        for g_id, max_counts in maxes()
        if max_counts["red"] <= 12
        and max_counts["green"] <= 13
        and max_counts["blue"] <= 14
    )


def p2(input: str):
    d = parse_input(input.splitlines())

    def maxes():
        for _, rounds in d:
            max_counts: dict[str, int] = {}
            for round in rounds:
                for color, count in round.items():
                    if color not in max_counts or max_counts[color] < count:
                        max_counts[color] = count
            yield max_counts

    return sum(
        max_counts["red"] * max_counts["green"] * max_counts["blue"]
        for max_counts in maxes()
    )
