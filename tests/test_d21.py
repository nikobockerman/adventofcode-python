from typing import reveal_type

import pytest

from adventofcode import d21

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
    map_ = d21._InfiniteMap(_example_input.splitlines())  # type: ignore # noqa: SLF001
    assert "".join(map_.get_row(2, 0, 10)) == rows[2]
    assert "".join(map_.get_row(2, -1, 1)) == rows[2][10:] + rows[2][:2]
    assert "".join(map_.get_row(2, 10, 12)) == rows[2][10:] + rows[2][:2]
    assert "".join(map_.get_row(2, 10, 23)) == rows[2][10:] + rows[2] + rows[2][:2]
    assert "".join(map_.get_row(2, -13, 24)) == rows[2][9:] + 3 * rows[2] + rows[2][:3]
