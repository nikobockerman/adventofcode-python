from __future__ import annotations

import math
from typing import TYPE_CHECKING, assert_never, final, overload

from .directions import CardinalDirection, CardinalDirectionsAll, RotationDirection

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


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

    def adjoins(self) -> Iterable[tuple[CardinalDirection, Coord2d]]:
        for direction in CardinalDirectionsAll:
            yield direction, self.adjoin(direction)

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

    def distance_to_cardinal(self, other: Coord2d) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def distance_to(self, other: Coord2d) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Coord2d({self.x}, {self.y})"


class Map2dEmptyDataError(ValueError):
    def __init__(self) -> None:
        super().__init__("data must not be empty")


class Map2dRectangularDataError(ValueError):
    def __init__(self) -> None:
        super().__init__("data must be rectangular")


class Map2d[Map2dDataType]:
    __slots__ = ("_sequence_data", "_width", "_height", "_last_x", "_last_y")

    def __init__(
        self,
        data: Iterable[Iterable[Map2dDataType]] | Iterable[Sequence[Map2dDataType]],
    ) -> None:
        self._sequence_data = tuple(tuple(row) for row in data)
        if len(self._sequence_data) == 0:
            raise Map2dEmptyDataError

        self._height = len(self._sequence_data)
        assert self._height > 0
        self._width = len(self._sequence_data[0])
        self._last_x = self._width - 1
        self._last_y = self._height - 1

        if not all(len(row) == self._width for row in self._sequence_data):
            raise Map2dRectangularDataError

        if self._width == 0:
            raise Map2dEmptyDataError

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def first_x(self) -> int:
        return 0

    @property
    def first_y(self) -> int:
        return 0

    @property
    def last_x(self) -> int:
        return self._last_x

    @property
    def last_y(self) -> int:
        return self._last_y

    def _get(self, x: int, y: int) -> Map2dDataType:
        if x < 0 or x > self._last_x or y < 0 or y > self._last_y:
            raise IndexError((x, y))
        return self._sequence_data[y][x]

    def _get_or_default(
        self, x: int, y: int, default: Map2dDataType | None = None
    ) -> Map2dDataType | None:
        if x < 0 or x > self._last_x or y < 0 or y > self._last_y:
            return default
        return self._get(x, y)

    @overload
    def get(self, coord: Coord2d, /) -> Map2dDataType: ...

    @overload
    def get(
        self, coord: Coord2d, /, default: Map2dDataType | None = None
    ) -> Map2dDataType | None: ...

    @overload
    def get(self, x_y: tuple[int, int], /) -> Map2dDataType: ...

    @overload
    def get(
        self, x_y: tuple[int, int], /, default: Map2dDataType | None = None
    ) -> Map2dDataType | None: ...

    def get(
        self,
        first: Coord2d | tuple[int, int],
        *args: Map2dDataType | None,
        **kwargs: Map2dDataType | None,
    ) -> Map2dDataType | None:
        if isinstance(first, Coord2d):
            x: int = first.x
            y: int = first.y
        else:
            x = first[0]
            y = first[1]

        if len(args) > 1:
            raise TypeError(args)

        if args:
            return self._get_or_default(x, y, args[0])
        if "default" in kwargs:
            return self._get_or_default(x, y, kwargs["default"])
        return self._get(x, y)

    def get_by_xy(self, x: int, y: int) -> Map2dDataType:
        return self._get(x, y)

    @final
    def __iter_data_by_lines(  # noqa: PLR0912, optimized for performance
        # so can't reduce branching here which is done for sanitizing input values
        self,
        first_x: int,
        first_y: int,
        last_x: int,
        last_y: int,
    ) -> Iterable[tuple[int, Iterable[tuple[int, Map2dDataType]]]]:
        step_x = 1 if first_x <= last_x else -1
        step_y = 1 if first_y <= last_y else -1
        if first_x < 0:
            first_x = 0
        elif first_x > self._last_x:
            first_x = self._last_x
        if last_x < 0:
            last_x = 0
        elif last_x > self._last_x:
            last_x = self._last_x
        if first_y < 0:
            first_y = 0
        elif first_y > self._last_y:
            first_y = self._last_y
        if last_y <= 0:
            slice_rows = self._sequence_data[first_y::step_y]
        else:
            if last_y < 0:
                last_y = 0
            elif last_y > self._last_y:
                last_y = self._last_y
            slice_rows = self._sequence_data[first_y : last_y + step_y : step_y]

        for row_ind, row in enumerate(slice_rows):
            y = row_ind * step_y + first_y
            if last_x <= 0:
                slice_row_datas = row[first_x::step_x]
            else:
                slice_row_datas = row[first_x : last_x + step_x : step_x]

            yield (
                y,
                (
                    (x_ind * step_x + first_x, data)
                    for x_ind, data in enumerate(slice_row_datas)
                ),
            )

    @final
    def __iter_data_by_columns(
        self, first_x: int, first_y: int, last_x: int, last_y: int
    ) -> Iterable[tuple[int, Iterable[tuple[int, Map2dDataType]]]]:
        step_x = 1 if first_x <= last_x else -1
        step_y = 1 if first_y <= last_y else -1
        if first_y < 0:
            first_y = 0
        elif first_y >= self._height:
            first_y = self._height - 1
        if last_y < 0:
            last_y = 0
        elif last_y >= self._height:
            last_y = self._height - 1
        for x in range(first_x, last_x + step_x, step_x):
            if x < 0 or x >= self._width:
                continue

            yield (
                x,
                (
                    (y, self.__get(x, y))
                    for y in range(first_y, last_y + step_y, step_y)
                ),
            )

    def iter_data(
        self,
        first_corner: Coord2d | None = None,
        last_corner: Coord2d | None = None,
        *,
        columns_first: bool = False,
    ) -> Iterable[tuple[int, Iterable[tuple[int, Map2dDataType]]]]:
        if first_corner is None:
            first_x = -1
            first_y = -1
        else:
            first_x = first_corner.x
            first_y = first_corner.y

        if last_corner is None:
            last_x = self._width
            last_y = self._height
        else:
            last_x = last_corner.x
            last_y = last_corner.y

        if first_x < 0 and last_x < 0:
            return
        if first_x > self.last_x and last_x >= self.last_x:
            return
        if first_y < 0 and last_y < 0:
            return
        if first_y > self.last_y and last_y >= self.last_y:
            return

        if not columns_first:
            yield from self.__iter_data_by_lines(first_x, first_y, last_x, last_y)
        else:
            yield from self.__iter_data_by_columns(first_x, first_y, last_x, last_y)

    def str_lines(
        self, get_symbol: Callable[[Map2dDataType], str] | None = None
    ) -> Iterable[str]:
        def default_get_symbol(x: Map2dDataType) -> str:
            return str(x)

        if get_symbol is None:
            get_symbol = default_get_symbol

        def row_symbols(row: Sequence[Map2dDataType]) -> Iterable[str]:
            for elem in row:
                sym = get_symbol(elem)
                assert len(sym) == 1
                yield sym

        for row in self._sequence_data:
            yield "".join(row_symbols(row))

    def __str__(self) -> str:
        return "\n".join(self.str_lines())

    def transpose(self) -> Map2d[Map2dDataType]:
        return Map2d(list(zip(*self._sequence_data, strict=True)))

    def __rotate_once_clockwise(self) -> Map2d[Map2dDataType]:
        return Map2d(
            (data for _, data in items)
            for _, items in self.__iter_data_by_columns(
                0, self._height - 1, self._width - 1, 0
            )
        )

    def __rotate_once_counterclockwise(self) -> Map2d[Map2dDataType]:
        return Map2d(
            (data for _, data in items)
            for _, items in self.__iter_data_by_columns(
                self._width - 1, 0, 0, self._height - 1
            )
        )

    def rotate(
        self, direction: RotationDirection, count: int = 1
    ) -> Map2d[Map2dDataType]:
        if count <= 0:
            raise ValueError(count)
        map_ = self
        for _ in range(count):
            if direction is RotationDirection.Clockwise:
                map_ = map_.__rotate_once_clockwise()  # noqa: SLF001
            elif direction is RotationDirection.Counterclockwise:
                map_ = map_.__rotate_once_counterclockwise()  # noqa: SLF001
            else:
                assert_never(direction)
        return map_

    def __hash__(self) -> int:
        return hash(self._sequence_data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self._sequence_data == other._sequence_data
        return NotImplemented
