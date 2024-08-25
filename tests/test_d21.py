from collections import Counter
from typing import reveal_type

import pytest

from adventofcode import d21
from adventofcode.tooling.map import Coord2d

_example_input = r"""
...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........
""".strip()


def test_p1() -> None:
    assert d21.p1(_example_input, 6) == 16


p2_cases = {
    1: 2,
    2: 4,
    3: 6,
    6: 16,
    10: 50,
    50: 1_594,
    100: 6_536,
    500: 167_004,
    1_000: 668_697,
    5_000: 16_733_044,
}


@pytest.mark.parametrize(("steps", "expected"), p2_cases.items(), ids=p2_cases.keys())
def test_p2(steps: int, expected: int) -> None:
    assert d21.p2(_example_input, steps) == expected


def test_infinite_map() -> None:
    rows = list(_example_input.splitlines())
    map_ = d21._InfiniteMap(_example_input.splitlines())  # noqa: SLF001
    assert "".join(map_.get_row(2, 0, 10)) == rows[2]
    assert "".join(map_.get_row(2, -1, 1)) == rows[2][10:] + rows[2][:2]
    assert "".join(map_.get_row(2, 10, 12)) == rows[2][10:] + rows[2][:2]
    assert "".join(map_.get_row(2, 10, 23)) == rows[2][10:] + rows[2] + rows[2][:2]
    assert "".join(map_.get_row(2, -13, 24)) == rows[2][9:] + 3 * rows[2] + rows[2][:3]

    map_data = _example_input.splitlines()
    for y, map_line in enumerate(map_data):
        if "S" in map_line:
            map_data[y] = map_line.replace("S", ".")

    for expected_y, (y, x_iter) in enumerate(
        map_.iter_data(Coord2d(0, 0), Coord2d(10, 10))
    ):
        assert y == expected_y
        assert "".join(x_data[1] for x_data in x_iter) == map_data[y]

    for expected_y, (y, x_iter) in enumerate(
        map_.iter_data(Coord2d(-11, -11), Coord2d(-1, -1))
    ):
        assert y + 11 == expected_y
        assert "".join(x_data[1] for x_data in x_iter) == map_data[y]


def test_area_square_full() -> None:
    full = d21._AreaSquareFull(  # noqa: SLF001
        Coord2d(0, 0),
        Coord2d(10, 10),
        1,
        5,
        10,
        Counter[int]({1: 6, 2: 4, 3: 2, 4: 1, 5: 6}),
    )
    assert full.count_possible_garden_plots_at_step(0) == 0
    assert full.count_possible_garden_plots_at_step(1) == 6
    assert full.count_possible_garden_plots_at_step(2) == 4
    assert full.count_possible_garden_plots_at_step(3) == 8
    assert full.count_possible_garden_plots_at_step(4) == 5
    assert full.count_possible_garden_plots_at_step(5) == 14
    assert full.count_possible_garden_plots_at_step(6) == 5
    assert full.count_possible_garden_plots_at_step(7) == 14
    assert full.count_possible_garden_plots_at_step(19) == 14
    assert full.count_possible_garden_plots_at_step(20) == 5


def test_area_square_extended() -> None:
    full = d21._AreaSquareFull(  # noqa: SLF001
        Coord2d(0, 0),
        Coord2d(10, 10),
        1,
        5,
        20,
        Counter[int]({1: 6, 2: 4, 3: 2, 4: 1, 5: 6}),
    )
    extended = d21._AreaSquareExtended(  # noqa: SLF001
        Coord2d(22, 22),
        Coord2d(32, 32),
        7,
        11,
        full,
    )

    assert extended.count_possible_garden_plots_at_step(12) == 5


def test_extendable_direction() -> None:
    dir = d21._ExtendableDirection(
        d21._AreaSquareFull(
            Coord2d(0, 0),
            Coord2d(10, 10),
            1,
            5,
            10,
            Counter[int]({1: 6, 2: 4, 3: 2, 4: 1, 5: 6}),
        ),
        d21._AreaSquareFull(
            Coord2d(11, 0),
            Coord2d(21, 10),
            4,
            8,
            10,
            Counter[int]({4: 2, 5: 3, 6: 5, 7: 2, 8: 1}),
        ),
        include_base_and_next=True,
    )
    area_results = list(dir.count_possible_garden_plots(10))
    assert area_results[0].location.top_left_coord == Coord2d(0, 0)
    assert area_results[1].location.top_left_coord == Coord2d(11, 0)
    assert area_results[2].location.top_left_coord == Coord2d(22, 0)
    assert area_results[3].location.top_left_coord == Coord2d(33, 0)
    assert len(area_results) == 4


def test_extendable_direction_2() -> None:
    dir = d21._ExtendableDirection(
        d21._AreaSquareFull(
            Coord2d(0, 0),
            Coord2d(10, 10),
            5,
            12,
            10,
            Counter[int]({1: 6, 2: 4, 3: 2, 4: 1, 5: 6}),
        ),
        d21._AreaSquareFull(
            Coord2d(11, 0),
            Coord2d(21, 10),
            8,
            15,
            10,
            Counter[int]({4: 2, 5: 3, 6: 5, 7: 2, 8: 1, 13: 1}),
        ),
        include_base_and_next=True,
    )
    area_results = list(dir.count_possible_garden_plots(10))
    assert area_results[0].location.top_left_coord == Coord2d(0, 0)
    assert area_results[1].location.top_left_coord == Coord2d(11, 0)
    # assert area_results[2].location.top_left_coord == Coord2d(22, 0)
    # assert area_results[3].location.top_left_coord == Coord2d(33, 0)
    assert len(area_results) == 2
