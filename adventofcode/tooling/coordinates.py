from __future__ import annotations

import math
from typing import assert_never

from adventofcode.tooling.directions import CardinalDirection


class Coord2d:
    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Coord2d):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def adjoin(self, direction: CardinalDirection) -> Coord2d:
        if direction is CardinalDirection.N:
            return Coord2d(self.x, self.y - 1)
        if direction is CardinalDirection.E:
            return Coord2d(self.x + 1, self.y)
        if direction is CardinalDirection.S:
            return Coord2d(self.x, self.y + 1)
        if direction is CardinalDirection.W:
            return Coord2d(self.x - 1, self.y)
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
            return Coord2d(self.x, self.y - distance)
        if direction is CardinalDirection.E:
            return Coord2d(self.x + distance, self.y)
        if direction is CardinalDirection.S:
            return Coord2d(self.x, self.y + distance)
        if direction is CardinalDirection.W:
            return Coord2d(self.x - distance, self.y)
        assert_never(direction)

    def distance_to_int(self, other: Coord2d) -> int:
        if self.y == other.y:
            return abs(self.x - other.x)
        if self.x == other.x:
            return abs(self.y - other.y)
        return math.isqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to(self, other: Coord2d) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Coord2d({self.x}, {self.y})"
