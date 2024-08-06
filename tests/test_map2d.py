from typing import Iterable, Sequence

from adventofcode.tooling.directions import RotationDirection
from adventofcode.tooling.map import (
    Coord2d,
    Map2d,
    Map2dEmptyDataError,
    Map2dRectangularDataError,
)


def test_init_empty() -> None:
    empty_iterables: Iterable[Iterable[Iterable[int]]] = ([], [[]], [[], []])
    for empty_iterable in empty_iterables:
        try:
            Map2d(empty_iterable)
        except Map2dEmptyDataError:
            pass
        else:
            raise AssertionError("Expected Map2dEmptyDataError")  # noqa: TRY003

        empty_iterable_seq: Iterable[Sequence[int]] = [list(x) for x in empty_iterable]
        try:
            Map2d(empty_iterable_seq)
        except Map2dEmptyDataError:
            pass
        else:
            raise AssertionError("Expected Map2dEmptyDataError")  # noqa: TRY003


def test_init_non_rectangular() -> None:
    non_square_iterables: Iterable[Iterable[Iterable[int]]] = (
        [[1, 2], [3, 4, 5]],
        [[1, 2, 3], [4, 5]],
    )
    for non_square_iterable in non_square_iterables:
        try:
            Map2d(non_square_iterable)
        except Map2dRectangularDataError:  # noqa: PERF203
            pass
        else:
            raise AssertionError("Expected Map2dRectangularDataError")  # noqa: TRY003


def test_transpose() -> None:
    map_ = Map2d(
        [
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["g", "h", "i"],
        ]
    )
    assert map_.transpose() == Map2d(
        [
            ["a", "d", "g"],
            ["b", "e", "h"],
            ["c", "f", "i"],
        ]
    )
    assert map_.transpose().transpose() == map_

    map_ = Map2d(
        [
            ["a", "b", "c"],
            ["d", "e", "f"],
        ]
    )
    assert map_.transpose() == Map2d(
        [
            ["a", "d"],
            ["b", "e"],
            ["c", "f"],
        ]
    )
    assert map_.transpose().transpose() == map_


def test_rotate():
    map_ = Map2d(
        [
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["g", "h", "i"],
        ]
    )
    assert map_.rotate(RotationDirection.Clockwise) == Map2d(
        [
            ["g", "d", "a"],
            ["h", "e", "b"],
            ["i", "f", "c"],
        ]
    )
    assert map_.rotate(RotationDirection.Clockwise, 2) == Map2d(
        [
            ["i", "h", "g"],
            ["f", "e", "d"],
            ["c", "b", "a"],
        ]
    )
    assert map_.rotate(RotationDirection.Clockwise, 3) == Map2d(
        [
            ["c", "f", "i"],
            ["b", "e", "h"],
            ["a", "d", "g"],
        ]
    )
    assert map_.rotate(RotationDirection.Clockwise, 4) == map_
    assert map_.rotate(RotationDirection.Counterclockwise, 1) == Map2d(
        [
            ["c", "f", "i"],
            ["b", "e", "h"],
            ["a", "d", "g"],
        ]
    )
    assert map_.rotate(RotationDirection.Counterclockwise, 2) == Map2d(
        [
            ["i", "h", "g"],
            ["f", "e", "d"],
            ["c", "b", "a"],
        ]
    )
    assert map_.rotate(RotationDirection.Counterclockwise, 3) == Map2d(
        [
            ["g", "d", "a"],
            ["h", "e", "b"],
            ["i", "f", "c"],
        ]
    )
    assert map_.rotate(RotationDirection.Counterclockwise, 4) == map_


def test_iter_data_full():
    map_ = Map2d(
        [
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["g", "h", "i"],
        ]
    )
    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data()
    ) == [
        (0, [(0, "a"), (1, "b"), (2, "c")]),
        (1, [(0, "d"), (1, "e"), (2, "f")]),
        (2, [(0, "g"), (1, "h"), (2, "i")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(columns_first=True)
    ) == [
        (0, [(0, "a"), (1, "d"), (2, "g")]),
        (1, [(0, "b"), (1, "e"), (2, "h")]),
        (2, [(0, "c"), (1, "f"), (2, "i")]),
    ]

    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data(
            Coord2d(map_.last_x, map_.last_y), Coord2d(0, 0)
        )
    ) == [
        (2, [(2, "i"), (1, "h"), (0, "g")]),
        (1, [(2, "f"), (1, "e"), (0, "d")]),
        (0, [(2, "c"), (1, "b"), (0, "a")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(
            Coord2d(map_.last_x, map_.last_y), Coord2d(0, 0), columns_first=True
        )
    ) == [
        (2, [(2, "i"), (1, "f"), (0, "c")]),
        (1, [(2, "h"), (1, "e"), (0, "b")]),
        (0, [(2, "g"), (1, "d"), (0, "a")]),
    ]

    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data(
            Coord2d(0, map_.last_y), Coord2d(map_.last_x, 0)
        )
    ) == [
        (2, [(0, "g"), (1, "h"), (2, "i")]),
        (1, [(0, "d"), (1, "e"), (2, "f")]),
        (0, [(0, "a"), (1, "b"), (2, "c")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(
            Coord2d(0, map_.last_y), Coord2d(map_.last_x, 0), columns_first=True
        )
    ) == [
        (0, [(2, "g"), (1, "d"), (0, "a")]),
        (1, [(2, "h"), (1, "e"), (0, "b")]),
        (2, [(2, "i"), (1, "f"), (0, "c")]),
    ]

    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data(
            Coord2d(map_.last_x, 0), Coord2d(0, map_.last_y)
        )
    ) == [
        (0, [(2, "c"), (1, "b"), (0, "a")]),
        (1, [(2, "f"), (1, "e"), (0, "d")]),
        (2, [(2, "i"), (1, "h"), (0, "g")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(
            Coord2d(map_.last_x, 0), Coord2d(0, map_.last_y), columns_first=True
        )
    ) == [
        (2, [(0, "c"), (1, "f"), (2, "i")]),
        (1, [(0, "b"), (1, "e"), (2, "h")]),
        (0, [(0, "a"), (1, "d"), (2, "g")]),
    ]


def test_iter_data_partial():
    map_ = Map2d(
        [
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
            ["i", "j", "k", "l"],
            ["m", "n", "o", "p"],
        ]
    )

    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data(Coord2d(1, 1), Coord2d(2, 2))
    ) == [
        (1, [(1, "f"), (2, "g")]),
        (2, [(1, "j"), (2, "k")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(
            Coord2d(1, 1), Coord2d(2, 2), columns_first=True
        )
    ) == [
        (1, [(1, "f"), (2, "j")]),
        (2, [(1, "g"), (2, "k")]),
    ]

    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data(Coord2d(2, 2), Coord2d(1, 1))
    ) == [
        (2, [(2, "k"), (1, "j")]),
        (1, [(2, "g"), (1, "f")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(
            Coord2d(2, 2), Coord2d(1, 1), columns_first=True
        )
    ) == [
        (2, [(2, "k"), (1, "g")]),
        (1, [(2, "j"), (1, "f")]),
    ]

    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data(Coord2d(1, 2), Coord2d(2, 1))
    ) == [
        (2, [(1, "j"), (2, "k")]),
        (1, [(1, "f"), (2, "g")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(
            Coord2d(1, 2), Coord2d(2, 1), columns_first=True
        )
    ) == [
        (1, [(2, "j"), (1, "f")]),
        (2, [(2, "k"), (1, "g")]),
    ]

    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data(Coord2d(2, 1), Coord2d(1, 2))
    ) == [
        (1, [(2, "g"), (1, "f")]),
        (2, [(2, "k"), (1, "j")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(
            Coord2d(2, 1), Coord2d(1, 2), columns_first=True
        )
    ) == [
        (2, [(1, "g"), (2, "k")]),
        (1, [(1, "f"), (2, "j")]),
    ]


def test_iter_data_completely_outside():
    map_ = Map2d(
        [
            ["a", "b"],
            ["c", "d"],
        ]
    )

    def verify_empty_iter_data(corner1: Coord2d, corner2: Coord2d):
        assert list(map_.iter_data(corner1, corner2)) == []
        assert list(map_.iter_data(corner1, corner2, columns_first=True)) == []
        assert list(map_.iter_data(corner2, corner1)) == []
        assert list(map_.iter_data(corner2, corner1, columns_first=True)) == []

    verify_empty_iter_data(Coord2d(0, -2), Coord2d(map_.last_x, -1))
    verify_empty_iter_data(Coord2d(-2, 0), Coord2d(-1, map_.last_y))
    verify_empty_iter_data(
        Coord2d(map_.last_x + 2, 0), Coord2d(map_.last_x + 1, map_.last_y)
    )
    verify_empty_iter_data(
        Coord2d(0, map_.last_y + 2), Coord2d(map_.last_x, map_.last_y + 1)
    )


def test_iter_partially_outside_left_top_corner():
    map_ = Map2d(
        [
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["g", "h", "i"],
        ]
    )

    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data(Coord2d(-1, -1), Coord2d(1, 1))
    ) == [
        (0, [(0, "a"), (1, "b")]),
        (1, [(0, "d"), (1, "e")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(
            Coord2d(-1, -1), Coord2d(1, 1), columns_first=True
        )
    ) == [
        (0, [(0, "a"), (1, "d")]),
        (1, [(0, "b"), (1, "e")]),
    ]

    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data(Coord2d(1, 1), Coord2d(-1, -1))
    ) == [
        (1, [(1, "e"), (0, "d")]),
        (0, [(1, "b"), (0, "a")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(
            Coord2d(1, 1), Coord2d(-1, -1), columns_first=True
        )
    ) == [
        (1, [(1, "e"), (0, "b")]),
        (0, [(1, "d"), (0, "a")]),
    ]


def test_iter_partially_outside_right_top_corner():
    map_ = Map2d(
        [
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["g", "h", "i"],
        ]
    )

    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data(Coord2d(1, -1), Coord2d(map_.last_x + 1, 1))
    ) == [
        (0, [(1, "b"), (2, "c")]),
        (1, [(1, "e"), (2, "f")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(
            Coord2d(1, -1), Coord2d(map_.last_x + 1, 1), columns_first=True
        )
    ) == [
        (1, [(0, "b"), (1, "e")]),
        (2, [(0, "c"), (1, "f")]),
    ]

    assert list(
        (y, list((x, data) for x, data in data_iter))
        for y, data_iter in map_.iter_data(Coord2d(1, 1), Coord2d(-1, -1))
    ) == [
        (1, [(1, "e"), (0, "d")]),
        (0, [(1, "b"), (0, "a")]),
    ]
    assert list(
        (x, list((y, data) for y, data in data_iter))
        for x, data_iter in map_.iter_data(
            Coord2d(1, 1), Coord2d(-1, -1), columns_first=True
        )
    ) == [
        (1, [(1, "e"), (0, "b")]),
        (0, [(1, "d"), (0, "a")]),
    ]
