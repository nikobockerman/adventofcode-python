from adventofcode.tooling.directions import CardinalDirection as Dir
from adventofcode.tooling.map import Coord2d


def test_comparison():
    assert Coord2d(3, 3) == Coord2d(3, 3)
    assert Coord2d(1, 2) == Coord2d(1, 2)
    assert Coord2d(1, 2) != Coord2d(2, 1)
    assert Coord2d(2, 2) != Coord2d(1, 1)


def test_hash():
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


def test_adjoin():
    assert Coord2d(1, 1).adjoin(Dir.N) == Coord2d(1, 0)
    assert Coord2d(1, 1).adjoin(Dir.E) == Coord2d(2, 1)
    assert Coord2d(1, 1).adjoin(Dir.S) == Coord2d(1, 2)
    assert Coord2d(1, 1).adjoin(Dir.W) == Coord2d(0, 1)


def test_dir_to():
    assert Coord2d(1, 1).dir_to(Coord2d(1, 0)) == Dir.N
    assert Coord2d(1, 1).dir_to(Coord2d(2, 1)) == Dir.E
    assert Coord2d(1, 1).dir_to(Coord2d(1, 2)) == Dir.S
    assert Coord2d(1, 1).dir_to(Coord2d(0, 1)) == Dir.W

    assert Coord2d(1, 1).dir_to(Coord2d(2, 2)) in (Dir.E, Dir.S)
    assert Coord2d(1, 1).dir_to(Coord2d(0, 2)) in (Dir.W, Dir.S)
    assert Coord2d(1, 1).dir_to(Coord2d(2, 0)) in (Dir.E, Dir.N)
    assert Coord2d(1, 1).dir_to(Coord2d(0, 0)) in (Dir.W, Dir.N)

    try:
        Coord2d(1, 1).dir_to(Coord2d(1, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")  # noqa: TRY003
