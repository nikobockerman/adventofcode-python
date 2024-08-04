from typing import Iterable, Literal, Sequence, get_args

from adventofcode.tooling.map import AllDirections, Coord2d, Map2d

_Rock = Literal["#"]
_GardenPlot = Literal["."]
_Start = Literal["S"]

_MapInputSymbol = _Rock | _GardenPlot | _Start
_MapSymbol = _Rock | _GardenPlot


def _parse_symbol(symbol: str) -> _MapInputSymbol:
    accepted_types = get_args(_MapInputSymbol)
    for accepted_symbols in map(get_args, accepted_types):
        assert len(accepted_symbols) == 1
        accepted_symbol = accepted_symbols[0]
        if symbol == accepted_symbol:
            return accepted_symbol
    raise _InvalidSymbolError(symbol)


class _InvalidSymbolError(ValueError):
    def __init__(self, symbol: str) -> None:
        super().__init__(f"Invalid symbol: {symbol}")


class _Map(Map2d[_MapSymbol]):
    def __init__(self, data: Iterable[Sequence[str]]) -> None:
        start: Coord2d | None = None
        start_symbol_count = 0
        map_data: list[list[_MapSymbol]] = []
        for line_index, line in enumerate(data):
            line_data: list[_MapSymbol] = []
            for index, symbol in enumerate(map(_parse_symbol, line)):
                if symbol == "S":
                    start = Coord2d(index, line_index)
                    start_symbol_count += 1
                    line_data.append(".")
                else:
                    line_data.append(symbol)
            map_data.append(line_data)
        assert start_symbol_count == 1
        assert start is not None

        super().__init__(map_data)
        self._start: Coord2d = start

    @property
    def start(self) -> Coord2d:
        return self._start


def p1(input_str: str) -> int:
    map_ = _Map(input_str.splitlines())

    locations_before_step: set[Coord2d] = {map_.start}
    for _ in range(64):
        locations_after_step: set[Coord2d] = set()
        for location in locations_before_step:
            for direction in AllDirections:
                new_location = location.adjoin(direction)
                try:
                    if map_.get(new_location) == "#":
                        continue
                    locations_after_step.add(new_location)
                except IndexError:
                    continue
        locations_before_step = locations_after_step
    return len(locations_before_step)
