from adventofcode.tooling.directions import CardinalDirection, RotationDirection


def test_cardinal_directions_str():
    assert str(CardinalDirection.N) == "n"
    assert str(CardinalDirection.E) == "e"
    assert str(CardinalDirection.S) == "s"
    assert str(CardinalDirection.W) == "w"


def test_rotation_directions_str():
    assert str(RotationDirection.Clockwise) == "clockwise"
    assert str(RotationDirection.Counterclockwise) == "counterclockwise"


def test_cardinal_directions_rotate():
    assert (
        CardinalDirection.N.rotate(RotationDirection.Clockwise) == CardinalDirection.E
    )
    assert (
        CardinalDirection.N.rotate(RotationDirection.Counterclockwise)
        == CardinalDirection.W
    )
    assert (
        CardinalDirection.E.rotate(RotationDirection.Clockwise) == CardinalDirection.S
    )
    assert (
        CardinalDirection.E.rotate(RotationDirection.Counterclockwise)
        == CardinalDirection.N
    )
    assert (
        CardinalDirection.S.rotate(RotationDirection.Clockwise) == CardinalDirection.W
    )
    assert (
        CardinalDirection.S.rotate(RotationDirection.Counterclockwise)
        == CardinalDirection.E
    )
    assert (
        CardinalDirection.W.rotate(RotationDirection.Clockwise) == CardinalDirection.N
    )
    assert (
        CardinalDirection.W.rotate(RotationDirection.Counterclockwise)
        == CardinalDirection.S
    )


def test_cardinal_directions_opposite():
    assert CardinalDirection.N.opposite() == CardinalDirection.S
    assert CardinalDirection.E.opposite() == CardinalDirection.W
    assert CardinalDirection.S.opposite() == CardinalDirection.N
    assert CardinalDirection.W.opposite() == CardinalDirection.E
