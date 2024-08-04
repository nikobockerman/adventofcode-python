from __future__ import annotations

import collections.abc
import itertools
import logging
import typing
from collections import Counter
from dataclasses import dataclass
from typing import (
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Sequence,
    assert_never,
    get_args,
)

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

    def infinite_get(self, x: int, y: int) -> _MapSymbol:
        x_ = x % self.width
        y_ = y % self.height
        return self.get((x_, y_))

    def get_row(self, y: int, min_x: int, max_x: int) -> tuple[_MapSymbol, ...]:
        y_ = y % self.height
        start_x = min_x % self.width
        stop_x = max_x % self.width
        if min_x < 0 or self.width <= max_x:
            data = list[_MapSymbol]()
            data.extend(self._sequence_data[y_][start_x:])
            while min_x < max_x - self.width:
                data.extend(self._sequence_data[y_])
                min_x += self.width
            data.extend(self._sequence_data[y_][: stop_x + 1])
            data = tuple(data)
        else:
            data = self._sequence_data[y_][start_x : stop_x + 1]
        return data


class _MapRow:
    def __init__(self, data: tuple[_MapSymbol, ...], start_x: int) -> None:
        self.data = data
        self.start_x = start_x

    def __getitem__(self, x: int) -> _MapSymbol:
        return self.data[x - self.start_x]

    def __repr__(self) -> str:
        return f"MapRow({self.data!r}, {self.start_x})"


class _VisitCounts:
    def __init__(self) -> None:
        self.visit_counts = Counter[int]()

    def add(self, step: int) -> None:
        self.visit_counts.update((step,))


@dataclass
class _LocationToProcessData:
    map_rows: Sequence[_MapRow]
    visited: Sequence[set[_LocationToProcess]]
    skipped: Sequence[set[int]]

    def __post_init__(self) -> None:
        assert len(self.map_rows) == 3
        assert len(self.visited) == 3
        assert len(self.skipped) == 3


class _LocationToProcess:
    def __init__(self, y: int, x: int, step: int) -> None:
        self.y = y
        self.x = x
        self.step = step

    def _process_neighbor(
        self,
        x: int,
        y: int,
        data_index: int,
        data: _LocationToProcessData,
        visit_counts: _VisitCounts,
    ) -> _LocationToProcess | None:
        if x in [x_.x for x_ in data.visited[data_index]]:
            return None
        symbol = data.map_rows[data_index][x]
        if symbol == "#":
            data.skipped[data_index].add(x)
            return None

        if symbol != ".":
            assert_never(symbol)

        step_new = self.step + 1
        data.visited[data_index].add(_LocationToProcess(y, x, step_new))
        visit_counts.add(step_new)
        return _LocationToProcess(y, x, step_new)

    def process_neighbors(
        self, data: _LocationToProcessData, visit_counts: _VisitCounts
    ) -> Iterator[_LocationToProcess]:
        neighbor = self._process_neighbor(self.x - 1, self.y, 1, data, visit_counts)
        if neighbor is not None:
            yield neighbor

        neighbor = self._process_neighbor(self.x + 1, self.y, 1, data, visit_counts)
        if neighbor is not None:
            yield neighbor

        neighbor = self._process_neighbor(self.x, self.y - 1, 0, data, visit_counts)
        if neighbor is not None:
            yield neighbor

        neighbor = self._process_neighbor(self.x, self.y + 1, 2, data, visit_counts)
        if neighbor is not None:
            yield neighbor


def p2(input_str: str, steps: int = 26_501_365) -> int:
    map_ = _InfiniteMap(input_str.splitlines())

    if steps > 500:
        steps = 500

    # if step % log_output_interval == 0 or step == steps - 1:  # noqa: SIM102
    #        if _logger.isEnabledFor(logging.DEBUG):
    #            _logger.info(
    #                "Step %d: visited=%d, skipped=%d, processing: %d",
    #                step,
    #                len(visited_locations),
    #                len(skipped_locations),
    #                len(locations_to_process),
    #            )

    # log_output_interval = steps // 10 if steps >= 10 else 1

    visited_locations: dict[int, set[_LocationToProcess]] = {
        map_.start.y: {_LocationToProcess(map_.start.y, map_.start.x, 0)}
    }
    skipped_locations: dict[int, set[int]] = dict()
    visit_counts = _VisitCounts()
    visit_counts.add(0)

    def get_create_row_set[T](collection: dict[int, set[T]], y: int) -> set[T]:
        result = collection.get(y)
        if result is None:
            collection[y] = result = set[T]()
        return result

    def get_create_row_list[T](collection: dict[int, list[T]], y: int) -> list[T]:
        result = collection.get(y)
        if result is None:
            collection[y] = result = list[T]()
        return result

    def get_related_rows[T](collection: dict[int, set[T]], y: int) -> list[set[T]]:
        return [get_create_row_set(collection, y_) for y_ in range(y - 1, y + 2)]

    def get_process_data(
        locations_to_process: list[_LocationToProcess],
    ) -> _LocationToProcessData:
        assert locations_to_process
        min_x: int | None = None
        max_x: int | None = None
        y = locations_to_process[0].y
        for loc_to_process in locations_to_process:
            assert loc_to_process.y == y
            if min_x is None or loc_to_process.x < min_x:
                min_x = loc_to_process.x
            if max_x is None or loc_to_process.x > max_x:
                max_x = loc_to_process.x
        assert min_x is not None
        assert max_x is not None
        return _LocationToProcessData(
            map_rows=[
                _MapRow(map_.get_row(y_, min_x - 1, max_x + 1), min_x - 1)
                for y_ in range(y - 1, y + 2)
            ],
            visited=get_related_rows(visited_locations, y),
            skipped=get_related_rows(skipped_locations, y),
        )

    to_process: dict[int, list[_LocationToProcess]] = {
        map_.start.y: [_LocationToProcess(map_.start.y, map_.start.x, 0)]
    }
    while to_process:
        rows_to_process = list(to_process.keys())
        for y in rows_to_process:
            locations_to_process = to_process.pop(y)
            process_data = get_process_data(locations_to_process)

            new_locations_to_process: list[_LocationToProcess] = []
            for loc_to_process in locations_to_process:
                for new_loc_to_process in loc_to_process.process_neighbors(
                    process_data, visit_counts
                ):
                    if new_loc_to_process.step >= steps:
                        continue
                    new_locations_to_process.append(new_loc_to_process)

            for y_, loc_to_processes in itertools.groupby(
                sorted(new_locations_to_process, key=lambda x: x.y), lambda x: x.y
            ):
                to_process_row = get_create_row_list(to_process, y_)
                # to_process_row.extend(loc_to_processes)
                locs = list(loc_to_processes)
                to_process_row.extend(locs)
                for loc in locs:
                    loc_x = loc.x
                    loc_y = loc.y
                    loc_x_map = loc.x % map_.width
                    loc_y_map = loc.y % map_.height
                    symbol_loc_map = map_.infinite_get(loc_x_map, loc_y_map)
                    symbol_loc = map_.infinite_get(loc_x, loc_y)
                    assert symbol_loc == "."
                    assert symbol_loc_map == "."

    def _map_str(visited: dict[int, set[_LocationToProcess]]) -> Iterator[str]:
        all_visited = set((Coord2d(x.x, y) for y, xs in visited.items() for x in xs))
        min_x = min(x.x for x in all_visited)
        max_x = max(x.x for x in all_visited)
        min_y = min(x.y for x in all_visited)
        max_y = max(x.y for x in all_visited)
        for y in range(min_y, max_y + 1):
            row = ""
            for x in range(min_x, max_x + 1):
                coord = Coord2d(x, y)
                if coord in all_visited:
                    row += "O"
                else:
                    row += "."
            yield row

    _logger.info("Visited: \n%s", "\n".join(_map_str(visited_locations)))

    visited_counts = {
        step: len(list(locs))
        for step, locs in itertools.groupby(
            sorted(
                (x for xs in visited_locations.values() for x in xs),
                key=lambda x: x.step,
            ),
            key=lambda x: x.step,
        )
    }
    visit_counts_ = visit_counts.visit_counts

    def _counts_str(c: dict[int, int]) -> Iterator[str]:
        for step, count in sorted(c.items()):
            yield f"{step}: {count}"

    _logger.info("Visited counts: %s", ", ".join(_counts_str(visited_counts)))
    _logger.info("Visit counts  : %s", ", ".join(_counts_str(visit_counts_)))

    return sum(
        count
        for step, count in visit_counts.visit_counts.items()
        if step % 2 == steps % 2
    )
