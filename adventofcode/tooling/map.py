from enum import Enum, auto
from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar, overload


class Dir(Enum):
    N = auto()
    E = auto()
    S = auto()
    W = auto()

    def turn_left(self) -> "Dir":
        if self is Dir.N:
            return Dir.W
        if self is Dir.E:
            return Dir.N
        if self is Dir.S:
            return Dir.E
        if self is Dir.W:
            return Dir.S
        raise ValueError(self)

    def turn_right(self) -> "Dir":
        if self is Dir.N:
            return Dir.E
        if self is Dir.E:
            return Dir.S
        if self is Dir.S:
            return Dir.W
        if self is Dir.W:
            return Dir.N
        raise ValueError(self)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


class RotationDir(Enum):
    Clockwise = auto()
    CW = Clockwise
    Counterclockwise = auto()
    CCW = Counterclockwise

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


AllDirections = (Dir.N, Dir.E, Dir.S, Dir.W)


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

    def adjoin(self, direction: Dir) -> "Coord2d":
        if direction is Dir.N:
            return Coord2d(self.x, self.y - 1)
        if direction is Dir.E:
            return Coord2d(self.x + 1, self.y)
        if direction is Dir.S:
            return Coord2d(self.x, self.y + 1)
        if direction is Dir.W:
            return Coord2d(self.x - 1, self.y)
        raise ValueError(direction)

    def dir_to(self, other: "Coord2d") -> Dir:
        if other.x > self.x:
            return Dir.E
        if other.x < self.x:
            return Dir.W
        if other.y > self.y:
            return Dir.S
        if other.y < self.y:
            return Dir.N
        raise ValueError(other)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


class Map2dEmptyDataError(ValueError):
    def __init__(self) -> None:
        super().__init__("data must not be empty")


class Map2dRectangularDataError(ValueError):
    def __init__(self) -> None:
        super().__init__("data must be rectangular")


Map2dDataType = TypeVar("Map2dDataType")


class Map2d(Generic[Map2dDataType]):
    __slots__ = ("_sequence_data", "_width", "_height", "_last_x", "_last_y")

    def __init__(
        self,
        data: Iterable[Iterable[Map2dDataType]] | Iterable[Sequence[Map2dDataType]],
    ):
        if not data:
            raise Map2dEmptyDataError()
        self._sequence_data = tuple(tuple(row) for row in data)
        self._height = len(self._sequence_data)
        assert self._height > 0
        self._width = len(self._sequence_data[0])
        self._last_x = self._width - 1
        self._last_y = self._height - 1

        if not all(len(row) == self._width for row in self._sequence_data):
            raise Map2dRectangularDataError()

        if self._width == 0:
            raise Map2dEmptyDataError()

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

    def __get(self, x: int, y: int) -> Map2dDataType:
        return self._sequence_data[y][x]

    def __get_or_default(
        self, x: int, y: int, default: Map2dDataType | None = None
    ) -> Map2dDataType | None:
        if x < 0 or x > self._last_x or y < 0 or y > self._last_y:
            return default
        return self.__get(x, y)

    @overload
    def get(
        self, coord: Coord2d, /, default: Map2dDataType | None = None
    ) -> Map2dDataType | None:
        ...

    @overload
    def get(
        self, x: int, y: int, /, default: Map2dDataType | None = None
    ) -> Map2dDataType | None:
        ...

    def get(
        self,
        first: Coord2d | int,
        *args: Any,
        **kwargs: Map2dDataType | None,
    ) -> Map2dDataType | None:
        if isinstance(first, Coord2d):
            if len(args) > 1:
                raise TypeError(args)
            coord: Coord2d = first
            x: int = coord.x
            y: int = coord.y
            args_after_coord = args
        else:
            if len(args) == 0:
                raise TypeError(args)
            if not isinstance(args[0], int):
                raise TypeError(args[0])
            x = first
            y = args[0]
            args_after_coord = args[1:]

        if len(args_after_coord) > 1:
            raise TypeError(args_after_coord)
        if len(args_after_coord) == 0:
            default = kwargs.pop("default", None)
        else:
            default = args_after_coord[0]

        return self.__get_or_default(x, y, default)

    def __iter_data_by_lines(
        self, first_x: int, first_y: int, last_x: int, last_y: int
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
        if last_y < 0:
            last_y = 0
        elif last_y > self._last_y:
            last_y = self._last_y
        if last_y == 0:
            slice_rows = self._sequence_data[first_y::step_y]
        else:
            slice_rows = self._sequence_data[first_y : last_y + step_y : step_y]

        for row_ind, row in enumerate(slice_rows):
            y = row_ind * step_y + first_y
            if last_x <= 0:
                slice_row_datas = row[first_x::step_x]
            else:
                slice_row_datas = row[first_x : last_x + step_x : step_x]

            yield y, (
                (x_ind * step_x + first_x, data)
                for x_ind, data in enumerate(slice_row_datas)
            )

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

            yield x, (
                (y, self.__get(x, y)) for y in range(first_y, last_y + step_y, step_y)
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

    def transpose(self) -> "Map2d[Map2dDataType]":
        return Map2d(list(zip(*self._sequence_data)))

    def __rotate_once_clockwise(self) -> "Map2d[Map2dDataType]":
        return Map2d(
            (data for _, data in items)
            for _, items in self.__iter_data_by_columns(
                0, self._height - 1, self._width - 1, 0
            )
        )

    def __rotate_once_counterclockwise(self) -> "Map2d[Map2dDataType]":
        return Map2d(
            (data for _, data in items)
            for _, items in self.__iter_data_by_columns(
                self._width - 1, 0, 0, self._height - 1
            )
        )

    def rotate(self, direction: RotationDir, count: int = 1) -> "Map2d[Map2dDataType]":
        if count <= 0:
            raise ValueError(count)
        map_ = self
        for _ in range(count):
            if direction is RotationDir.Clockwise:
                map_ = map_.__rotate_once_clockwise()  # noqa: SLF001
            elif direction is RotationDir.Counterclockwise:
                map_ = map_.__rotate_once_counterclockwise()  # noqa: SLF001
            else:
                raise ValueError(direction)
        return map_

    def __hash__(self) -> int:
        return hash(self._sequence_data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self._sequence_data == other._sequence_data
        return NotImplemented
