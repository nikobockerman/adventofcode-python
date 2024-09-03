import itertools
import logging
from collections.abc import Iterable

from adventofcode.tooling.coordinates import X, Y
from adventofcode.tooling.map import Map2d


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


def _compare_datas(
    data1: Iterable[str], data2: Iterable[str], allowed_mismatches: int
) -> int | None:
    mismatches = 0
    for sym1, sym2 in zip(data1, data2, strict=False):
        if sym1 != sym2:
            mismatches += 1
            if mismatches > allowed_mismatches:
                return None
    return mismatches


def _find_consecutive_rows_or_columns(
    map_: Map2d[str], start_pos: int, find_column_not_row: bool, allowed_mismatches: int
) -> tuple[int, int] | None:
    if find_column_not_row:
        first_y = Y(0)
        first_x = X(start_pos)
    else:
        first_y = Y(start_pos)
        first_x = X(0)
    for (i1, data1), (_, data2) in itertools.pairwise(
        (i, [sym for _, sym in sym_iter])
        for i, sym_iter in map_.iter_data(
            first_y, first_x, columns_first=find_column_not_row
        )
    ):
        match_res = _compare_datas(data1, data2, allowed_mismatches)
        if match_res is None:
            continue
        return i1, match_res
    return None


def _map_data_iter_to_data(d: tuple[int, str]) -> str:
    return d[1]


def _check_if_datas_around_reflection_match(
    map_: Map2d[str],
    pos_before_reflection: int,
    find_column_not_row: bool,
    allowed_mismatches: int,
) -> int | None:
    if find_column_not_row:
        before_first_y = Y(0)
        before_first_x = X(pos_before_reflection - 1)
        before_last_y = map_.br_y
        before_last_x = X(-1)
        after_first_y = Y(0)
        after_first_x = X(pos_before_reflection + 2)
    else:
        before_first_y = Y(pos_before_reflection - 1)
        before_first_x = X(0)
        before_last_y = Y(-1)
        before_last_x = map_.br_x
        after_first_y = Y(pos_before_reflection + 2)
        after_first_x = X(0)

    mismatches = 0

    for (i1, data1_iter), (i2, data2_iter) in zip(
        map_.iter_data(before_first_y, before_first_x, before_last_y, before_last_x),
        map_.iter_data(after_first_y, after_first_x),
        strict=False,
    ):
        match_res = _compare_datas(
            map(_map_data_iter_to_data, data1_iter),
            map(_map_data_iter_to_data, data2_iter),
            allowed_mismatches,
        )
        if match_res is None:
            logging.debug("Invalid mirror data at %d and %d", i1, i2)
            return None

        assert match_res >= 0
        mismatches += match_res
        if mismatches > allowed_mismatches:
            return None

    return mismatches


def _find_reflection_line(
    map_: Map2d[str], find_column_not_row: bool, required_mismatches: int = 0
) -> int | None:
    logging.debug(
        "Searching for reflection with %d required mismatches", required_mismatches
    )
    search_start_pos = 0
    while True:
        remaining_mismatches = required_mismatches
        found_data_info = _find_consecutive_rows_or_columns(
            map_, search_start_pos, find_column_not_row, remaining_mismatches
        )
        if found_data_info is None:
            logging.debug(
                "No consecutive datas found with %d allowed mismatches",
                remaining_mismatches,
            )
            return None

        first_pos, mismatches = found_data_info
        search_start_pos = first_pos + 1
        remaining_mismatches -= mismatches

        logging.debug(
            "Found reflection at %d with %d mismatches remaining",
            first_pos,
            remaining_mismatches,
        )

        check_res = _check_if_datas_around_reflection_match(
            map_, first_pos, find_column_not_row, remaining_mismatches
        )
        if check_res is None:
            logging.debug(
                "Datas around reflection don't match with %d allowed mismatches",
                remaining_mismatches,
            )
            continue

        remaining_mismatches -= check_res
        assert remaining_mismatches >= 0
        if remaining_mismatches > 0:
            logging.debug("Not enough mismatches")
            continue
        logging.debug("Found perfect reflection at %d", first_pos)
        return first_pos


def _resolve(input_str: str, required_mismatches_per_map: int) -> int:
    result: int = 0
    for map_counter, map_ in enumerate(_parse_maps(input_str), 1):
        logging.info(
            "Map %2d: Size (LxC) %2d x %2d\n%s",
            map_counter,
            map_.height,
            map_.width,
            map_,
        )
        match_index = _find_reflection_line(map_, False, required_mismatches_per_map)
        if match_index is not None:
            line_or_column = "L"
            match_multiplier = 100
        else:
            match_index = _find_reflection_line(map_, True, required_mismatches_per_map)
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
