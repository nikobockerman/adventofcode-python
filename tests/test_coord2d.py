import re

import pytest

from adventofcode.tooling.coordinates import Coord2d, X, Y
from adventofcode.tooling.directions import CardinalDirection as Dir


def test_comparison() -> None:
    assert Coord2d(Y(3), X(3)) == Coord2d(Y(3), X(3))
    assert Coord2d(Y(2), X(1)) == Coord2d(Y(2), X(1))
    assert Coord2d(Y(2), X(1)) != Coord2d(Y(1), X(2))
    assert Coord2d(Y(2), X(2)) != Coord2d(Y(1), X(1))


def test_hash() -> None:
    assert hash(Coord2d(Y(1), X(1))) == hash(Coord2d(Y(1), X(1)))
    assert hash(Coord2d(Y(2), X(1))) == hash(Coord2d(Y(2), X(1)))
    assert hash(Coord2d(Y(1), X(2))) == hash(Coord2d(Y(1), X(2)))
    assert hash(Coord2d(Y(2), X(2))) == hash(Coord2d(Y(2), X(2)))
    assert hash(Coord2d(Y(1), X(1))) != hash(Coord2d(Y(2), X(1)))
    assert hash(Coord2d(Y(1), X(1))) != hash(Coord2d(Y(1), X(2)))
    assert hash(Coord2d(Y(1), X(1))) != hash(Coord2d(Y(2), X(2)))
    assert hash(Coord2d(Y(2), X(1))) != hash(Coord2d(Y(1), X(2)))
    assert hash(Coord2d(Y(2), X(1))) != hash(Coord2d(Y(2), X(2)))
    assert hash(Coord2d(Y(1), X(2))) != hash(Coord2d(Y(2), X(2)))


def test_adjoin() -> None:
    assert Coord2d(Y(1), X(1)).adjoin(Dir.N) == Coord2d(Y(0), X(1))
    assert Coord2d(Y(1), X(1)).adjoin(Dir.E) == Coord2d(Y(1), X(2))
    assert Coord2d(Y(1), X(1)).adjoin(Dir.S) == Coord2d(Y(2), X(1))
    assert Coord2d(Y(1), X(1)).adjoin(Dir.W) == Coord2d(Y(1), X(0))


def test_dir_to() -> None:
    assert Coord2d(Y(1), X(1)).dir_to(Coord2d(Y(0), X(1))) == Dir.N
    assert Coord2d(Y(1), X(1)).dir_to(Coord2d(Y(1), X(2))) == Dir.E
    assert Coord2d(Y(1), X(1)).dir_to(Coord2d(Y(2), X(1))) == Dir.S
    assert Coord2d(Y(1), X(1)).dir_to(Coord2d(Y(1), X(0))) == Dir.W

    assert Coord2d(Y(1), X(1)).dir_to(Coord2d(Y(2), X(2))) in (Dir.E, Dir.S)
    assert Coord2d(Y(1), X(1)).dir_to(Coord2d(Y(2), X(0))) in (Dir.W, Dir.S)
    assert Coord2d(Y(1), X(1)).dir_to(Coord2d(Y(0), X(2))) in (Dir.E, Dir.N)
    assert Coord2d(Y(1), X(1)).dir_to(Coord2d(Y(0), X(0))) in (Dir.W, Dir.N)

    other = Coord2d(Y(1), X(1))
    with pytest.raises(ValueError, match=re.escape(str(other))):
        Coord2d(Y(1), X(1)).dir_to(other)
