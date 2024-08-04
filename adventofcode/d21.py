import logging
from typing import Iterable, Iterator, Literal, Sequence, assert_never, get_args

from adventofcode.tooling.map import AllDirections, Coord2d, Map2d

_logger = logging.getLogger(__name__)

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


def p1(input_str: str, steps: int = 64) -> int:
    map_ = _Map(input_str.splitlines())

    locations_before_step: set[Coord2d] = {map_.start}
    for _ in range(steps):
        locations_after_step: set[Coord2d] = set()
        for location in locations_before_step:
            for direction in AllDirections:
                new_location = location.adjoin(direction)
                try:
                    symbol = map_.get(new_location)
                except IndexError:
                    continue
                if symbol == "#":
                    continue
                if symbol == ".":
                    locations_after_step.add(new_location)
                else:
                    assert_never(symbol)
        locations_before_step = locations_after_step
    return len(locations_before_step)


class _InfiniteMap(Map2d[_MapSymbol]):
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

    def infinite_get(self, coord: Coord2d) -> _MapSymbol:
        x = coord.x % self.width
        y = coord.y % self.height
        return self.get((x, y))


def p2(input_str: str, steps: int = 26_501_365) -> int:
    map_ = _InfiniteMap(input_str.splitlines())

    if steps > 500:
        steps = 500

    log_output_interval = steps // 10 if steps >= 10 else 1
    visited_locations: set[Coord2d] = {map_.start}
    skipped_locations: set[Coord2d] = set()
    visit_counts: dict[int, int] = {0: 1}

    locations_to_process: set[Coord2d] = {map_.start}
    for step in range(steps):
        if step % log_output_interval == 0 or step == steps - 1:
            _logger.info(
                "Step %d: visited=%d, skipped=%d, processing: %d",
                step,
                len(visited_locations),
                len(skipped_locations),
                len(locations_to_process),
            )
        new_visits: int = 0
        new_locations: set[Coord2d] = set()
        for prev_location in locations_to_process:
            for direction in AllDirections:
                new_location = prev_location.adjoin(direction)
                if new_location in skipped_locations:
                    continue
                symbol = map_.infinite_get(new_location)
                if symbol == "#":
                    skipped_locations.add(new_location)
                    continue
                if symbol == ".":
                    if new_location in visited_locations:
                        continue
                    visited_locations.add(new_location)
                    new_visits += 1
                    new_locations.add(new_location)
                else:
                    assert_never(symbol)
        locations_to_process = new_locations
        visit_counts[step + 1] = new_visits

    def _map_str(visited: set[Coord2d], height: int, width: int) -> Iterator[str]:
        for y in range(height):
            row = ""
            for x in range(width):
                coord = Coord2d(x, y)
                if coord in visited:
                    row += "O"
                else:
                    row += "."
            yield row

    _logger.info(
        "Visited: \n%s", "\n".join(_map_str(visited_locations, map_.height, map_.width))
    )

    return sum(count for step, count in visit_counts.items() if step % 2 == steps % 2)
