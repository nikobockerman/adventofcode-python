import pytest

from adventofcode.tooling.coordinates import Coord2d
from adventofcode.tooling.directions import CardinalDirection as Dir


def test_comparison() -> None:
    assert Coord2d(3, 3) == Coord2d(3, 3)
    assert Coord2d(1, 2) == Coord2d(1, 2)
    assert Coord2d(1, 2) != Coord2d(2, 1)
    assert Coord2d(2, 2) != Coord2d(1, 1)


def test_hash() -> None:
    assert hash(Coord2d(1, 1)) == hash(Coord2d(1, 1))
    assert hash(Coord2d(1, 2)) == hash(Coord2d(1, 2))
    assert hash(Coord2d(2, 1)) == hash(Coord2d(2, 1))
    assert hash(Coord2d(2, 2)) == hash(Coord2d(2, 2))
    assert hash(Coord2d(1, 1)) != hash(Coord2d(1, 2))
    assert hash(Coord2d(1, 1)) != hash(Coord2d(2, 1))
    assert hash(Coord2d(1, 1)) != hash(Coord2d(2, 2))
    assert hash(Coord2d(1, 2)) != hash(Coord2d(2, 1))
    assert hash(Coord2d(1, 2)) != hash(Coord2d(2, 2))
    assert hash(Coord2d(2, 1)) != hash(Coord2d(2, 2))


def test_adjoin() -> None:
    assert Coord2d(1, 1).adjoin(Dir.N) == Coord2d(1, 0)
    assert Coord2d(1, 1).adjoin(Dir.E) == Coord2d(2, 1)
    assert Coord2d(1, 1).adjoin(Dir.S) == Coord2d(1, 2)
    assert Coord2d(1, 1).adjoin(Dir.W) == Coord2d(0, 1)


def test_dir_to() -> None:
    assert Coord2d(1, 1).dir_to(Coord2d(1, 0)) == Dir.N
    assert Coord2d(1, 1).dir_to(Coord2d(2, 1)) == Dir.E
    assert Coord2d(1, 1).dir_to(Coord2d(1, 2)) == Dir.S
    assert Coord2d(1, 1).dir_to(Coord2d(0, 1)) == Dir.W

    assert Coord2d(1, 1).dir_to(Coord2d(2, 2)) in (Dir.E, Dir.S)
    assert Coord2d(1, 1).dir_to(Coord2d(0, 2)) in (Dir.W, Dir.S)
    assert Coord2d(1, 1).dir_to(Coord2d(2, 0)) in (Dir.E, Dir.N)
    assert Coord2d(1, 1).dir_to(Coord2d(0, 0)) in (Dir.W, Dir.N)

    other = Coord2d(1, 1)
    with pytest.raises(ValueError, match=str(other)):
        Coord2d(1, 1).dir_to(other)
