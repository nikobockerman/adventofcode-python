import pytest

from adventofcode.tooling.coordinates import X, Y
from adventofcode.tooling.directions import RotationDirection
from adventofcode.tooling.map import (
    Map2d,
    Map2dEmptyDataError,
    Map2dRectangularDataError,
)


def test_init_empty() -> None:
    empty_sequences: tuple[list[list[int]], ...] = ([], [[]], [[], []])
    for empty_sequence in empty_sequences:
        it = iter(empty_sequence)
        with pytest.raises(Map2dEmptyDataError):
            Map2d(it)

        it_of_iterables = (iter(x) for x in empty_sequence)
        with pytest.raises(Map2dEmptyDataError):
            Map2d(it_of_iterables)


def test_init_non_rectangular() -> None:
    non_square_iterables = (
        [[1, 2], [3, 4, 5]],
        [[1, 2, 3], [4, 5]],
    )
    for non_square_iterable in non_square_iterables:
        with pytest.raises(Map2dRectangularDataError):
            Map2d(non_square_iterable)


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


def test_rotate() -> None:
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


def test_iter_data() -> None:
    map_ = Map2d(
        [
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["g", "h", "i"],
        ]
    )
    assert [(y, list(data_iter)) for y, data_iter in map_.iter_data()] == [
        (0, [(0, "a"), (1, "b"), (2, "c")]),
        (1, [(0, "d"), (1, "e"), (2, "f")]),
        (2, [(0, "g"), (1, "h"), (2, "i")]),
    ]
    assert [
        (x, list(data_iter)) for x, data_iter in map_.iter_data(columns_first=True)
    ] == [
        (0, [(0, "a"), (1, "d"), (2, "g")]),
        (1, [(0, "b"), (1, "e"), (2, "h")]),
        (2, [(0, "c"), (1, "f"), (2, "i")]),
    ]

    assert [
        (y, list(data_iter))
        for y, data_iter in map_.iter_data(map_.br_y, map_.br_x, Y(0), X(0))
    ] == [
        (2, [(2, "i"), (1, "h"), (0, "g")]),
        (1, [(2, "f"), (1, "e"), (0, "d")]),
        (0, [(2, "c"), (1, "b"), (0, "a")]),
    ]
    assert [
        (x, list(data_iter))
        for x, data_iter in map_.iter_data(
            map_.br_y, map_.br_x, Y(0), X(0), columns_first=True
        )
    ] == [
        (2, [(2, "i"), (1, "f"), (0, "c")]),
        (1, [(2, "h"), (1, "e"), (0, "b")]),
        (0, [(2, "g"), (1, "d"), (0, "a")]),
    ]

    assert [
        (y, list(data_iter))
        for y, data_iter in map_.iter_data(map_.br_y, X(0), Y(0), map_.br_x)
    ] == [
        (2, [(0, "g"), (1, "h"), (2, "i")]),
        (1, [(0, "d"), (1, "e"), (2, "f")]),
        (0, [(0, "a"), (1, "b"), (2, "c")]),
    ]
    assert [
        (x, list(data_iter))
        for x, data_iter in map_.iter_data(
            map_.br_y, X(0), Y(0), map_.br_x, columns_first=True
        )
    ] == [
        (0, [(2, "g"), (1, "d"), (0, "a")]),
        (1, [(2, "h"), (1, "e"), (0, "b")]),
        (2, [(2, "i"), (1, "f"), (0, "c")]),
    ]

    assert [
        (y, list(data_iter))
        for y, data_iter in map_.iter_data(Y(0), map_.br_x, map_.br_y, X(0))
    ] == [
        (0, [(2, "c"), (1, "b"), (0, "a")]),
        (1, [(2, "f"), (1, "e"), (0, "d")]),
        (2, [(2, "i"), (1, "h"), (0, "g")]),
    ]
    assert [
        (x, list(data_iter))
        for x, data_iter in map_.iter_data(
            Y(0), map_.br_x, map_.br_y, X(0), columns_first=True
        )
    ] == [
        (2, [(0, "c"), (1, "f"), (2, "i")]),
        (1, [(0, "b"), (1, "e"), (2, "h")]),
        (0, [(0, "a"), (1, "d"), (2, "g")]),
    ]


def test_iter_data_partial() -> None:
    map_ = Map2d(
        [
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
            ["i", "j", "k", "l"],
            ["m", "n", "o", "p"],
        ]
    )

    assert [
        (y, list(data_iter)) for y, data_iter in map_.iter_data(Y(1), X(1), Y(2), X(2))
    ] == [
        (1, [(1, "f"), (2, "g")]),
        (2, [(1, "j"), (2, "k")]),
    ]
    assert [
        (x, list(data_iter))
        for x, data_iter in map_.iter_data(Y(1), X(1), Y(2), X(2), columns_first=True)
    ] == [
        (1, [(1, "f"), (2, "j")]),
        (2, [(1, "g"), (2, "k")]),
    ]

    assert [
        (y, list(data_iter)) for y, data_iter in map_.iter_data(Y(2), X(2), Y(1), X(1))
    ] == [
        (2, [(2, "k"), (1, "j")]),
        (1, [(2, "g"), (1, "f")]),
    ]
    assert [
        (x, list(data_iter))
        for x, data_iter in map_.iter_data(Y(2), X(2), Y(1), X(1), columns_first=True)
    ] == [
        (2, [(2, "k"), (1, "g")]),
        (1, [(2, "j"), (1, "f")]),
    ]

    assert [
        (y, list(data_iter)) for y, data_iter in map_.iter_data(Y(2), X(1), Y(1), X(2))
    ] == [
        (2, [(1, "j"), (2, "k")]),
        (1, [(1, "f"), (2, "g")]),
    ]
    assert [
        (x, list(data_iter))
        for x, data_iter in map_.iter_data(Y(2), X(1), Y(1), X(2), columns_first=True)
    ] == [
        (1, [(2, "j"), (1, "f")]),
        (2, [(2, "k"), (1, "g")]),
    ]

    assert [
        (y, list(data_iter)) for y, data_iter in map_.iter_data(Y(1), X(2), Y(2), X(1))
    ] == [
        (1, [(2, "g"), (1, "f")]),
        (2, [(2, "k"), (1, "j")]),
    ]
    assert [
        (x, list(data_iter))
        for x, data_iter in map_.iter_data(Y(1), X(2), Y(2), X(1), columns_first=True)
    ] == [
        (2, [(1, "g"), (2, "k")]),
        (1, [(1, "f"), (2, "j")]),
    ]


def test_iter_data_completely_outside() -> None:
    map_ = Map2d(
        [
            ["a", "b"],
            ["c", "d"],
        ]
    )

    def verify_empty_iter_data(
        corner1_y: Y, corner1_x: X, corner2_y: Y, corner2_x: X
    ) -> None:
        assert list(map_.iter_data(corner1_y, corner1_x, corner2_y, corner2_x)) == []
        assert (
            list(
                map_.iter_data(
                    corner1_y, corner1_x, corner2_y, corner2_x, columns_first=True
                )
            )
            == []
        )
        assert list(map_.iter_data(corner2_y, corner2_x, corner1_y, corner1_x)) == []
        assert (
            list(
                map_.iter_data(
                    corner2_y, corner2_x, corner1_y, corner1_x, columns_first=True
                )
            )
            == []
        )

    verify_empty_iter_data(Y(-2), X(0), Y(-1), map_.br_x)
    verify_empty_iter_data(Y(0), X(-2), map_.br_y, X(-1))
    verify_empty_iter_data(Y(0), X(map_.br_x + 2), map_.br_y, X(map_.br_x + 1))
    verify_empty_iter_data(Y(map_.br_y + 2), X(0), Y(map_.br_y + 1), map_.br_x)


def test_iter_partially_outside_left_top_corner() -> None:
    map_ = Map2d(
        [
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["g", "h", "i"],
        ]
    )

    assert [
        (y, list(data_iter))
        for y, data_iter in map_.iter_data(Y(-1), X(-1), Y(1), X(1))
    ] == [
        (0, [(0, "a"), (1, "b")]),
        (1, [(0, "d"), (1, "e")]),
    ]
    assert [
        (x, list(data_iter))
        for x, data_iter in map_.iter_data(Y(-1), X(-1), Y(1), X(1), columns_first=True)
    ] == [
        (0, [(0, "a"), (1, "d")]),
        (1, [(0, "b"), (1, "e")]),
    ]

    assert [
        (y, list(data_iter))
        for y, data_iter in map_.iter_data(Y(1), X(1), Y(-1), X(-1))
    ] == [
        (1, [(1, "e"), (0, "d")]),
        (0, [(1, "b"), (0, "a")]),
    ]
    assert [
        (x, list(data_iter))
        for x, data_iter in map_.iter_data(Y(1), X(1), Y(-1), X(-1), columns_first=True)
    ] == [
        (1, [(1, "e"), (0, "b")]),
        (0, [(1, "d"), (0, "a")]),
    ]


def test_iter_partially_outside_right_top_corner() -> None:
    map_ = Map2d(
        [
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["g", "h", "i"],
        ]
    )

    assert [
        (y, list(data_iter))
        for y, data_iter in map_.iter_data(Y(-1), X(1), Y(1), X(map_.br_x + 1))
    ] == [
        (0, [(1, "b"), (2, "c")]),
        (1, [(1, "e"), (2, "f")]),
    ]
    assert [
        (x, list(data_iter))
        for x, data_iter in map_.iter_data(
            Y(-1), X(1), Y(1), X(map_.br_x + 1), columns_first=True
        )
    ] == [
        (1, [(0, "b"), (1, "e")]),
        (2, [(0, "c"), (1, "f")]),
    ]

    assert [
        (y, list(data_iter))
        for y, data_iter in map_.iter_data(Y(1), X(1), Y(-1), X(-1))
    ] == [
        (1, [(1, "e"), (0, "d")]),
        (0, [(1, "b"), (0, "a")]),
    ]
    assert [
        (x, list(data_iter))
        for x, data_iter in map_.iter_data(Y(1), X(1), Y(-1), X(-1), columns_first=True)
    ] == [
        (1, [(1, "e"), (0, "b")]),
        (0, [(1, "d"), (0, "a")]),
    ]
