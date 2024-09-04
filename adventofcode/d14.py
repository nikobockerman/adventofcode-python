import logging
from typing import TYPE_CHECKING, assert_never, cast

from adventofcode.tooling.coordinates import Coord2d, X, Y
from adventofcode.tooling.directions import CardinalDirection as Dir
from adventofcode.tooling.map import Map2d

if TYPE_CHECKING:
    from collections.abc import Iterable

_logger = logging.getLogger(__name__)


def _calculate_load(map_: Map2d[str]) -> int:
    max_load = map_.height
    return sum(
        max_load - y
        for y, x_iter in map_.iter_data()
        for _, sym in x_iter
        if sym == "O"
    )


def _roll_rocks(map_: Map2d[str], direction: Dir) -> Map2d[str]:
    lines: list[list[str]] = [["."] * map_.width for _ in range(map_.height)]

    def _coord_rows_first(outer: Y | X, inner: Y | X) -> Coord2d:
        return Coord2d(cast(Y, outer), cast(X, inner))

    def _coord_columns_first(outer: Y | X, inner: Y | X) -> Coord2d:
        return Coord2d(cast(Y, inner), cast(X, outer))

    map_iter: (
        Iterable[tuple[Y, Iterable[tuple[X, str]]]]
        | Iterable[tuple[X, Iterable[tuple[Y, str]]]]
    )
    if direction == Dir.N:
        map_iter = map_.iter_data(columns_first=True)
        coord_func = _coord_columns_first

        def set_rock(
            prev_square: Coord2d | None, rock_group_count: int, coord: Coord2d
        ) -> None:
            nonlocal lines
            y = (-1 if prev_square is None else prev_square.y) + rock_group_count
            lines[y][coord.x] = "O"

    elif direction == Dir.E:
        map_iter = map_.iter_data_by_lines(map_.tl_y, map_.br_x, map_.br_y, map_.tl_x)
        coord_func = _coord_rows_first

        def set_rock(
            prev_square: Coord2d | None, rock_group_count: int, coord: Coord2d
        ) -> None:
            nonlocal lines
            x = (
                map_.width if prev_square is None else prev_square.x
            ) - rock_group_count
            lines[coord.y][x] = "O"

    elif direction == Dir.S:
        map_iter = map_.iter_data_by_columns(map_.br_y, map_.tl_x, map_.tl_y, map_.br_x)
        coord_func = _coord_columns_first

        def set_rock(
            prev_square: Coord2d | None, rock_group_count: int, coord: Coord2d
        ) -> None:
            nonlocal lines
            y = (
                map_.height if prev_square is None else prev_square.y
            ) - rock_group_count
            lines[y][coord.x] = "O"

    elif direction == Dir.W:
        map_iter = map_.iter_data()
        coord_func = _coord_rows_first

        def set_rock(
            prev_square: Coord2d | None, rock_group_count: int, coord: Coord2d
        ) -> None:
            nonlocal lines
            x = (-1 if prev_square is None else prev_square.x) + rock_group_count
            lines[coord.y][x] = "O"

    else:
        assert_never(direction)

    for outer, data_iter in map_iter:
        prev_square: Coord2d | None = None
        rock_group_count = 0
        for inner, sym in data_iter:
            if sym == ".":
                continue
            coord = coord_func(outer, inner)
            if sym == "#":
                lines[coord.y][coord.x] = "#"
                prev_square = coord
                rock_group_count = 0
            elif sym == "O":
                rock_group_count += 1
                set_rock(prev_square, rock_group_count, coord)

    return Map2d(lines)


def p1(input_str: str) -> int:
    map_ = Map2d([list(line) for line in input_str.splitlines()])
    return _calculate_load(_roll_rocks(map_, Dir.N))


def _perform_spin(map_: Map2d[str]) -> Map2d[str]:
    for dir_ in (Dir.N, Dir.W, Dir.S, Dir.E):
        map_ = _roll_rocks(map_, dir_)
    return map_


def _get_rock_coords(map_: Map2d[str]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (x, y) for y, x_iter in map_.iter_data() for x, sym in x_iter if sym == "O"
    )


def p2(input_str: str) -> int:
    map_ = Map2d([list(line) for line in input_str.splitlines()])
    _logger.debug("Initial map:\n%s", map_)
    maps_after_spins: list[Map2d[str]] = []
    seen_rock_coords: dict[frozenset[tuple[int, int]], int] = {}
    final_map: Map2d[str] | None = None
    for i in range(1, 1_000_000_000 + 1):
        map_ = _perform_spin(map_)
        _logger.info("Done spinning %d", i)
        _logger.debug("Map after spin %d:\n%s", i, map_)
        rock_coords = _get_rock_coords(map_)
        seen = seen_rock_coords.get(rock_coords)
        if seen is not None:
            final_spin = seen + ((1_000_000_000 - seen) % (i - seen))
            _logger.info(
                "Found loop at %d matching spin %d -> final spin = %d",
                i,
                seen,
                final_spin,
            )
            final_map = maps_after_spins[final_spin - 1]
            break

        seen_rock_coords[rock_coords] = i
        maps_after_spins.append(map_)
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug("Load on spin %d: %d", i, _calculate_load(map_))

    assert final_map is not None
    _logger.debug("Final map:\n%s", final_map)
    return _calculate_load(final_map)
