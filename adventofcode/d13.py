import itertools
import logging
from typing import Iterable

from adventofcode.tooling.map import Coord2d, Map2d


def _parse_maps(input_str: str) -> Iterable[Map2d[str]]:
    lines: list[list[str]] = []
    for line in input_str.splitlines():
        if not line:
            yield Map2d(lines)
            lines = []
            continue
        lines.append(list(line))
    if lines:
        yield Map2d(lines)


def _compare_lines(
    line1: Iterable[str], line2: Iterable[str], allowed_mismatches: int
) -> int | None:
    mismatches = 0
    for sym1, sym2 in zip(line1, line2):
        if sym1 != sym2:
            mismatches += 1
            if mismatches > allowed_mismatches:
                return None
    return mismatches


def _find_consecutive_lines(
    map_: Map2d[str], start_line: int, allowed_mismatches: int
) -> tuple[int, int] | None:
    for (y1, y1_line), (_, y2_line) in itertools.islice(
        itertools.pairwise(
            (y, list(sym for _, sym in sym_iter)) for y, sym_iter in map_.iter_lines()
        ),
        start_line,
        None,
    ):
        match_res = _compare_lines(y1_line, y2_line, allowed_mismatches)
        if match_res is None:
            continue
        return y1, match_res
    return None


def _map_data_iter_to_data(d: tuple[Coord2d, str]) -> str:
    return d[1]


def _find_reflection_line(map_: Map2d[str], required_mismatches: int = 0) -> int | None:
    start = 0
    while True:
        remaining_mismatches = required_mismatches
        found_line_info = _find_consecutive_lines(map_, start, remaining_mismatches)
        if found_line_info is None:
            logging.debug(
                "No consecutive lines found with %d allowed mismatches",
                remaining_mismatches,
            )
            return None

        y_first, mismatches = found_line_info

        start = y_first + 1
        remaining_mismatches -= mismatches

        logging.debug(
            "Found consecutive lines at %d with %d mismatches. "
            "Remaining mismatches: %d",
            y_first,
            mismatches,
            remaining_mismatches,
        )

        for y in range(y_first - 1, -1, -1):
            y_below = y_first + 1 + (y_first - y)
            if y_below >= map_.len_y:
                logging.debug(
                    "No more lines below (%d + 1 - (%d - %d) = %d >= %d)",
                    y_first,
                    y_first,
                    y,
                    y_below,
                    map_.len_y,
                )
                continue

            match_res = _compare_lines(
                map(
                    _map_data_iter_to_data,
                    map_.iter_data(Coord2d(0, y), Coord2d(map_.len_x, y + 1)),
                ),
                map(
                    _map_data_iter_to_data,
                    map_.iter_data(
                        Coord2d(0, y_below), Coord2d(map_.len_x, y_below + 1)
                    ),
                ),
                remaining_mismatches,
            )
            if match_res is None:
                logging.debug(
                    "Invalid mirror line at y: %d and y_above: %d", y, y_below
                )
                break

            assert match_res >= 0
            assert match_res <= remaining_mismatches
            remaining_mismatches -= match_res
        else:
            assert remaining_mismatches >= 0
            if remaining_mismatches > 0:
                logging.debug("Not enough mismatches")
                continue
            logging.debug("Found perfect mirror line at y: %d", y_first)
            return y_first


def _resolve(input_str: str, required_mismatches_per_map: int) -> int:
    result: int = 0
    for map_counter, map_ in enumerate(_parse_maps(input_str), 1):
        logging.info(
            "Map %2d: Size (LxC) %2d x %2d\n%s",
            map_counter,
            map_.len_y,
            map_.len_x,
            map_,
        )
        match_index = _find_reflection_line(map_, required_mismatches_per_map)
        if match_index is not None:
            line_or_column = "L"
            match_multiplier = 100
        else:
            transposed_map = map_.transpose()
            logging.debug(
                "Transposed map %2d: Size (LxC) %2d x %2d\n%s",
                map_counter,
                transposed_map.len_y,
                transposed_map.len_x,
                transposed_map,
            )
            match_index = _find_reflection_line(
                transposed_map, required_mismatches_per_map
            )
            assert match_index is not None
            line_or_column = "C"
            match_multiplier = 1

        map_result = (match_index + 1) * match_multiplier
        result += map_result
        logging.info(
            "Map %2d: %s: %2d, map_result: %5d, result so far: %d",
            map_counter,
            line_or_column,
            match_index + 1,
            map_result,
            result,
        )

    return result


def p1(input_str: str) -> int:
    return _resolve(input_str, 0)


def p2(input_str: str) -> int:
    return _resolve(input_str, 1)
