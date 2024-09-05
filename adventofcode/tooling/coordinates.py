from __future__ import annotations

import math
from typing import NewType, assert_never

from adventofcode.tooling.directions import CardinalDirection

X = NewType("X", int)
Y = NewType("Y", int)


class Coord2d:
    __slots__ = ("y", "x")

    def __init__(self, y: Y, x: X) -> None:
        self.y = y
        self.x = x

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Coord2d):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def adjoin(self, direction: CardinalDirection) -> Coord2d:
        if direction is CardinalDirection.N:
            return Coord2d(Y(self.y - 1), self.x)
        if direction is CardinalDirection.E:
            return Coord2d(self.y, X(self.x + 1))
        if direction is CardinalDirection.S:
            return Coord2d(Y(self.y + 1), self.x)
        if direction is CardinalDirection.W:
            return Coord2d(self.y, X(self.x - 1))
        assert_never(direction)

    def dir_to(self, other: Coord2d) -> CardinalDirection:
        if other.x > self.x:
            return CardinalDirection.E
        if other.x < self.x:
            return CardinalDirection.W
        if other.y > self.y:
            return CardinalDirection.S
        if other.y < self.y:
            return CardinalDirection.N
        raise ValueError(other)

    def get_relative(self, direction: CardinalDirection, distance: int = 1) -> Coord2d:
        if direction is CardinalDirection.N:
            return Coord2d(Y(self.y - distance), self.x)
        if direction is CardinalDirection.E:
            return Coord2d(self.y, X(self.x + distance))
        if direction is CardinalDirection.S:
            return Coord2d(Y(self.y + distance), self.x)
        if direction is CardinalDirection.W:
            return Coord2d(self.y, X(self.x - distance))
        assert_never(direction)

    def distance_to_int(self, other: Coord2d) -> int:
        if self.y == other.y:
            return abs(self.x - other.x)
        if self.x == other.x:
            return abs(self.y - other.y)
        return math.isqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to(self, other: Coord2d) -> float:
        if self.y == other.y:
            return abs(self.x - other.x)
        if self.x == other.x:
            return abs(self.y - other.y)
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __str__(self) -> str:
        return f"({self.y}, {self.x})"

    def __repr__(self) -> str:
        return f"Coord2d({self.y}, {self.x})"


def adjoin_north(y: Y, x: X) -> tuple[Y, X]:
    return Y(y - 1), x


def adjoin_east(y: Y, x: X) -> tuple[Y, X]:
    return y, X(x + 1)


def adjoin_south(y: Y, x: X) -> tuple[Y, X]:
    return Y(y + 1), x


def adjoin_west(y: Y, x: X) -> tuple[Y, X]:
    return y, X(x - 1)


def adjoin_dir(y: Y, x: X, direction: CardinalDirection) -> tuple[Y, X]:
    if direction is CardinalDirection.N:
        return adjoin_north(y, x)
    if direction is CardinalDirection.E:
        return adjoin_east(y, x)
    if direction is CardinalDirection.S:
        return adjoin_south(y, x)
    if direction is CardinalDirection.W:
        return adjoin_west(y, x)
    assert_never(direction)
