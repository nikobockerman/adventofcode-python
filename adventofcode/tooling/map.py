from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Generic, Iterable, Sequence, TypeVar


class Dir(Enum):
    N = auto()
    E = auto()
    S = auto()
    W = auto()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


AllDirections = (Dir.N, Dir.E, Dir.S, Dir.W)


@dataclass(frozen=True)
class Coord2d:
    x: int
    y: int

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


class Map2dEmptyDataError(ValueError):
    def __init__(self) -> None:
        super().__init__("data must not be empty")


class Map2dRectangularDataError(ValueError):
    def __init__(self) -> None:
        super().__init__("data must be rectangular")


Map2dDataType = TypeVar("Map2dDataType")


class Map2d(Generic[Map2dDataType]):
    def __init__(self, data: Sequence[Sequence[Map2dDataType]]):
        if not data:
            raise Map2dEmptyDataError()
        self._len_y = len(data)
        self._len_x = len(data[0])

        if not all(len(row) == self._len_x for row in data):
            raise Map2dRectangularDataError()

        self._sequence_data = data
        self._dict_data = {
            Coord2d(x, y): data[y][x]
            for y in range(len(data))
            for x in range(len(data[y]))
        }

    @property
    def len_y(self) -> int:
        return self._len_y

    @property
    def len_x(self) -> int:
        return self._len_x

    def __get(self, coord: Coord2d) -> Map2dDataType:
        return self._dict_data[coord]

    def get(
        self, coord: Coord2d, default: Map2dDataType | None = None
    ) -> Map2dDataType | None:
        try:
            return self.__get(coord)
        except KeyError:
            return default

    def iter_data(
        self, start: Coord2d | None = None, stop: Coord2d | None = None
    ) -> Iterable[tuple[Coord2d, Map2dDataType]]:
        if start is None:
            start = Coord2d(0, 0)
        elif (
            start.x < 0 or start.y < 0 or start.x >= self.len_x or start.y >= self.len_y
        ):
            raise IndexError(start)

        if stop is None:
            stop = Coord2d(self.len_x, self.len_y)
        elif stop.x < 0 or stop.y < 0 or stop.x > self.len_x or stop.y > self.len_y:
            raise IndexError(stop)

        for y, row in enumerate(self._sequence_data[start.y : stop.y]):
            for x, data in enumerate(row[start.x : stop.x]):
                coord = Coord2d(x, y)
                yield coord, data

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
