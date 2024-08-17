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
    override,
)

from adventofcode.tooling.directions import CardinalDirectionsAll
from adventofcode.tooling.map import Coord2d, Map2d

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

    def get_adjoin_garden_plots_for_location(
        self, location: Coord2d
    ) -> Iterator[Coord2d]:
        for direction in CardinalDirectionsAll:
            new_location = location.adjoin(direction)
            try:
                symbol = self.get(new_location)
            except IndexError:
                continue

            if symbol == "#":
                continue

            if symbol == ".":
                yield new_location
            else:
                assert_never(symbol)

    def get_adjoin_garden_plots_for_locations(
        self, locations: Iterable[Coord2d]
    ) -> Iterator[Coord2d]:
        for location in locations:
            yield from self.get_adjoin_garden_plots_for_location(location)


def p1(input_str: str, steps: int = 64) -> int:
    map_ = _Map(input_str.splitlines())

    locations_to_process: set[Coord2d] = {map_.start}
    for _ in range(steps):
        locations_to_process = set(
            map_.get_adjoin_garden_plots_for_locations(locations_to_process)
        )
    return len(locations_to_process)


class _InfiniteMap(_Map):
    @property
    def start(self) -> Coord2d:
        return self._start

    def infinite_get(self, x: int, y: int) -> _MapSymbol:
        x_ = x % self.width
        y_ = y % self.height
        return self.get((x_, y_))

    def _iter_row(self, y: int, min_x: int, max_x: int) -> Iterator[_MapSymbol]:
        y_ = y % self.height
        start_x = min_x % self.width
        stop_x = max_x % self.width

        if min_x >= 0 and self.width > max_x:
            yield from self._sequence_data[y_][start_x : stop_x + 1]
            return

        yield from self._sequence_data[y_][start_x:]
        while min_x < max_x - self.width:
            yield from self._sequence_data[y_]
            min_x += self.width
        yield from self._sequence_data[y_][: stop_x + 1]

    def get_row(self, y: int, min_x: int, max_x: int) -> tuple[_MapSymbol, ...]:
        return tuple(self._iter_row(y, min_x, max_x))

    @override
    def iter_data(
        self,
        first_corner: Coord2d | None = None,
        last_corner: Coord2d | None = None,
        *,
        columns_first: bool = False,
    ) -> Iterable[tuple[int, Iterable[tuple[int, _MapSymbol]]]]:
        if columns_first is True:
            raise NotImplementedError()

        if first_corner is None or last_corner is None:
            raise NotImplementedError()

        if first_corner.x >= last_corner.x or first_corner.y >= last_corner.y:
            raise NotImplementedError()

        for y in range(first_corner.y, last_corner.y + 1):
            yield (
                y,
                zip(
                    range(first_corner.x, last_corner.x + 1),
                    self._iter_row(y, first_corner.x, last_corner.x),
                ),
            )


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
        # x=18 - -40 => -40+18=-22
        # y=72-5=67 - -40 => -40+67=27
        # ...
        # ..#
        # ..#
        ### Bug: First time visited with step == max_step-1 -> second visit with
        ### smaller step -> will not be processed further on first time and second time it gets skipped
        # visited_xs = sorted([x_.x for x_ in data.visited[data_index]])
        # if x in visited_xs:
        #    return None
        symbol = data.map_rows[data_index][x]
        if symbol == "#":
            data.skipped[data_index].add(x)
            return None

        if symbol != ".":
            assert_never(symbol)

        step_new = self.step + 1
        # data.visited[data_index].add(_LocationToProcess(y, x, step_new))
        # visit_counts.add(step_new)
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


@dataclass(slots=True)
class _AreaSquare:
    top_left_corner: Coord2d
    bottom_right_corner: Coord2d
    visits_at_step: dict[Coord2d, int]


@dataclass(slots=True)
class _ExtendableArea:
    base: _AreaSquare
    next_width: _AreaSquare
    next_height: _AreaSquare


@dataclass(slots=True)
class _CornerAreas:
    top_left: _ExtendableArea
    top_right: _ExtendableArea
    bottom_left: _ExtendableArea
    bottom_right: _ExtendableArea


def _p2_step1_resolve_around_start(map_: _InfiniteMap) -> _CornerAreas:
    # Validate assumptions
    assert map_.width % 2 == 1
    assert map_.height % 2 == 1
    assert map_.width // 2 == map_.height // 2
    assert map_.start.x == map_.width // 2
    assert map_.start.y == map_.height // 2

    @dataclass(slots=True)
    class _AreaRange:
        top_left: Coord2d
        bottom_right: Coord2d

        def width(self) -> int:
            return self.bottom_right.x - self.top_left.x + 1

        def height(self) -> int:
            return self.bottom_right.y - self.top_left.y + 1

        def contains(self, coord: Coord2d) -> bool:
            return (
                self.top_left.x <= coord.x <= self.bottom_right.x
                and self.top_left.y <= coord.y <= self.bottom_right.y
            )

        def to_square(self) -> _AreaSquare:
            return _AreaSquare(self.top_left, self.bottom_right, {})

    inclusive_width = map_.width - 1
    inclusive_height = map_.height - 1

    top_left = _AreaRange(
        Coord2d(map_.start.x - inclusive_width, map_.start.y - inclusive_height),
        Coord2d(map_.start.x, map_.start.y),
    )
    top_left_left = _AreaRange(
        Coord2d(top_left.top_left.x - map_.width, top_left.top_left.y),
        Coord2d(top_left.top_left.x - 1, top_left.bottom_right.y),
    )
    top_left_up = _AreaRange(
        Coord2d(top_left.top_left.x, top_left.top_left.y - map_.height),
        Coord2d(top_left.bottom_right.x, top_left.top_left.y - 1),
    )
    top_right = _AreaRange(
        Coord2d(map_.start.x, map_.start.y - inclusive_height),
        Coord2d(map_.start.x + inclusive_width, map_.start.y),
    )
    top_right_right = _AreaRange(
        Coord2d(top_right.bottom_right.x + 1, top_right.top_left.y),
        Coord2d(top_right.bottom_right.x + map_.width, top_right.bottom_right.y),
    )
    top_right_up = _AreaRange(
        Coord2d(top_right.top_left.x, top_right.top_left.y - map_.height),
        Coord2d(top_right.bottom_right.x, top_right.top_left.y - 1),
    )
    bottom_left = _AreaRange(
        Coord2d(map_.start.x - inclusive_width, map_.start.y),
        Coord2d(map_.start.x, map_.start.y + inclusive_height),
    )
    bottom_left_left = _AreaRange(
        Coord2d(bottom_left.top_left.x - map_.width, bottom_left.top_left.y),
        Coord2d(bottom_left.top_left.x - 1, bottom_left.bottom_right.y),
    )
    bottom_left_down = _AreaRange(
        Coord2d(bottom_left.top_left.x, bottom_left.bottom_right.y + 1),
        Coord2d(bottom_left.bottom_right.x, bottom_left.bottom_right.y + map_.height),
    )
    bottom_right = _AreaRange(
        Coord2d(map_.start.x, map_.start.y),
        Coord2d(map_.start.x + inclusive_width, map_.start.y + inclusive_height),
    )
    bottom_right_right = _AreaRange(
        Coord2d(bottom_right.bottom_right.x + 1, bottom_right.top_left.y),
        Coord2d(bottom_right.bottom_right.x + map_.width, bottom_right.bottom_right.y),
    )
    bottom_right_down = _AreaRange(
        Coord2d(bottom_right.top_left.x, bottom_right.bottom_right.y + 1),
        Coord2d(bottom_right.bottom_right.x, bottom_right.bottom_right.y + map_.height),
    )

    assert map_.width == top_left.width()
    assert map_.height == top_left.height()
    assert map_.width == top_right.width()
    assert map_.height == top_right.height()
    assert map_.width == bottom_left.width()
    assert map_.height == bottom_left.height()
    assert map_.width == bottom_right.width()
    assert map_.height == bottom_right.height()

    assert top_left.width() == top_left_left.width()
    assert top_left.height() == top_left_left.height()
    assert top_left.width() == top_left_up.width()
    assert top_left.height() == top_left_up.height()

    assert top_right.width() == top_right_right.width()
    assert top_right.height() == top_right_right.height()
    assert top_right.width() == top_right_up.width()
    assert top_right.height() == top_right_up.height()

    assert bottom_left.width() == bottom_left_left.width()
    assert bottom_left.height() == bottom_left_left.height()
    assert bottom_left.width() == bottom_left_down.width()
    assert bottom_left.height() == bottom_left_down.height()

    assert bottom_right.width() == bottom_right_right.width()
    assert bottom_right.height() == bottom_right_right.height()
    assert bottom_right.width() == bottom_right_down.width()
    assert bottom_right.height() == bottom_right_down.height()

    required_visited_locations = set[Coord2d](
        Coord2d(x, y)
        for area in (
            top_left,
            top_left_left,
            top_left_up,
            top_right,
            top_right_right,
            top_right_up,
            bottom_left,
            bottom_left_left,
            bottom_left_down,
            bottom_right,
            bottom_right_right,
            bottom_right_down,
        )
        for y, xs in map_.iter_data(area.top_left, area.bottom_right)
        for x, symbol in xs
        if symbol == "."
    )

    min_x = top_left_left.top_left.x
    max_x = top_right_right.bottom_right.x
    min_y = top_left_up.top_left.y
    max_y = bottom_right_down.bottom_right.y

    def _map_visited_str(visited: set[Coord2d]) -> Iterator[str]:
        for y in range(min_y, max_y + 1):
            row = ""
            for x in range(min_x, max_x + 1):
                coord = Coord2d(x, y)
                if coord in visited:
                    row += "O"
                else:
                    row += map_.infinite_get(x, y)
            yield row

    visited_locations_at_step: dict[int, set[Coord2d]] = {0: {map_.start}}
    visited_locations: set[Coord2d] = {map_.start}
    locations_to_process: set[Coord2d] = {map_.start}
    step = 0
    while not required_visited_locations <= visited_locations:
        step += 1
        adjoin_locations = set(
            map_.get_adjoin_garden_plots_for_locations(locations_to_process)
        )
        visited_locations_at_step[step] = adjoin_locations
        visited_locations.update(adjoin_locations)
        locations_to_process = adjoin_locations
        _logger.debug("Visited: \n%s", "\n".join(_map_visited_str(visited_locations)))

    areas = _CornerAreas(
        top_left=_ExtendableArea(
            base=top_left.to_square(),
            next_width=top_left_left.to_square(),
            next_height=top_left_up.to_square(),
        ),
        top_right=_ExtendableArea(
            base=top_right.to_square(),
            next_width=top_right_right.to_square(),
            next_height=top_right_up.to_square(),
        ),
        bottom_left=_ExtendableArea(
            base=bottom_left.to_square(),
            next_width=bottom_left_left.to_square(),
            next_height=bottom_left_down.to_square(),
        ),
        bottom_right=_ExtendableArea(
            base=bottom_right.to_square(),
            next_width=bottom_right_right.to_square(),
            next_height=bottom_right_down.to_square(),
        ),
    )

    area_pairs = list[tuple[_AreaRange, _AreaSquare]](
        (
            (top_left, areas.top_left.base),
            (top_left_left, areas.top_left.next_width),
            (top_left_up, areas.top_left.next_height),
            (top_right, areas.top_right.base),
            (top_right_right, areas.top_right.next_width),
            (top_right_up, areas.top_right.next_height),
            (bottom_left, areas.bottom_left.base),
            (bottom_left_left, areas.bottom_left.next_width),
            (bottom_left_down, areas.bottom_left.next_height),
            (bottom_right, areas.bottom_right.base),
            (bottom_right_right, areas.bottom_right.next_width),
            (bottom_right_down, areas.bottom_right.next_height),
        )
    )

    for step, visits in visited_locations_at_step.items():
        for coord in visits:
            for area_range, area_square in area_pairs:
                if area_range.contains(coord):
                    area_square.visits_at_step[coord] = step

    return areas


def _p2_step2_extend_areas(map_: _InfiniteMap, areas: _CornerAreas) -> None:
    # 2. Take the corner squares and extend them outbound from stars adjusting the step
    #    counts

    # Extend top-left towards top once (TODO: Keep extending)
    pass


# TODO
# 1. Process width-first from start until 4x4 squares with start in the middle have all
#    garden plots visited
# 2. Take the corner squares and extend them outbound from stars adjusting the step
#    counts
# 3. Keep extending until max step count in one extended square reaches steps, once that
#    happens ignore such square
# 4. Process the remaining outer bounds with the width-first algorighm
#    (or complex if too slow)


def p2(input_str: str, steps: int = 26_501_365) -> int:
    map_ = _InfiniteMap(input_str.splitlines())

    # Step 1: START
    corner_areas = _p2_step1_resolve_around_start(map_)
    # Step 1: END

    # Step 2: START
    _p2_step2_extend_areas(map_, corner_areas)
    # Step 2: END

    coords_and_steps = [
        (coord, step)
        for corner in (
            corner_areas.top_left,
            corner_areas.top_right,
            corner_areas.bottom_left,
            corner_areas.bottom_right,
        )
        for corner_square in (corner.base, corner.next_width, corner.next_height)
        for coord, step in corner_square.visits_at_step.items()
    ]

    coords_with_steps = dict[Coord2d, set[int]]()
    for coord, step in coords_and_steps:
        steps_ = coords_with_steps.setdefault(coord, set())
        steps_.add(step)
    for coord_, steps_ in coords_with_steps.items():
        if len(steps_) > 1:
            raise AssertionError(f"{coord_} found in more than one step: {steps_}")  # noqa: TRY003

    steps_remainder = steps % 2
    return sum(
        1
        for step_ in coords_with_steps.values()
        if next(iter(step_)) % 2 == steps_remainder
    )

    return 0

    if steps >= 100:
        log_output_interval = steps // 100
    elif steps >= 10:
        log_output_interval = steps // 10
    else:
        log_output_interval = 1
    # log_output_interval = steps // 10 if steps >= 10 else 1

    visited_locations: dict[int, dict[int, int]] = {map_.start.y: {map_.start.x: 0}}
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
            visited=[set(), set(), set()],  # get_related_rows(visited_locations, y),
            skipped=get_related_rows(skipped_locations, y),
        )

    count = 0
    next_print = 0
    to_process: dict[int, list[_LocationToProcess]] = {
        map_.start.y: [_LocationToProcess(map_.start.y, map_.start.x, 0)]
    }
    while to_process:
        # if step % log_output_interval == 0 or step == steps - 1:  # noqa: SIM102
        if count >= next_print:
            next_print += log_output_interval
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.info(
                    "Processed=%d, visited=%d, skipped=%d, max_step=%d",
                    count,
                    sum(len(x) for x in visited_locations.values()),
                    sum(len(x) for x in skipped_locations.values()),
                    max(max(x.values()) for x in visited_locations.values()),
                )
        rows_to_process = list(to_process.keys())
        for y in rows_to_process:
            locations_to_process = to_process.pop(y)
            process_data = get_process_data(locations_to_process)

            new_locations_to_process: list[_LocationToProcess] = []
            for loc_to_process in locations_to_process:
                for new_loc_to_process in loc_to_process.process_neighbors(
                    process_data, visit_counts
                ):
                    count += 1
                    if (
                        y_visited := visited_locations.get(new_loc_to_process.y)
                    ) is not None:
                        if (
                            earlier_step := y_visited.get(new_loc_to_process.x)
                        ) is not None and earlier_step <= new_loc_to_process.step:
                            continue
                        y_visited[new_loc_to_process.x] = new_loc_to_process.step
                    else:
                        visited_locations[new_loc_to_process.y] = {
                            new_loc_to_process.x: new_loc_to_process.step
                        }

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

    all_visited = set(
        (Coord2d(x, y) for y, xs in visited_locations.items() for x in xs.keys())
    )
    possible_steps = set(
        Coord2d(x, y)
        for y, xs in visited_locations.items()
        for x, step in xs.items()
        if step % 2 == steps_remainder
    )
    min_x = min(x.x for x in all_visited)
    max_x = max(x.x for x in all_visited)
    min_y = min(x.y for x in all_visited)
    max_y = max(x.y for x in all_visited)
    _logger.info("Min: %s, Max: %s", Coord2d(min_x, min_y), Coord2d(max_x, max_y))

    def _map_visited_str(visited: set[Coord2d]) -> Iterator[str]:
        for y in range(min_y, max_y + 1):
            row = ""
            for x in range(min_x, max_x + 1):
                coord = Coord2d(x, y)
                if coord in visited:
                    row += "O"
                else:
                    row += map_.infinite_get(x, y)
            yield row

    # _logger.info("Visited: \n%s", "\n".join(_map_visited_str()))
    with open(f"p21d2-{steps}-all-visited.txt", "w") as f:
        f.write("\n".join(_map_visited_str(all_visited)))
    with open(f"p21d2-{steps}-possible-visited.txt", "w") as f:
        f.write("\n".join(_map_visited_str(possible_steps)))

    # def _map_symbols() -> Iterator[str]:
    #    for y in range(min_y, max_y + 1):
    #        row = ""
    #        for x in range(min_x, max_x + 1):
    #            row += map_.infinite_get(x, y)
    #        yield f"{y:3}: {row}"

    # _logger.info("Map: \n%s", "\n".join(_map_symbols()))

    visited_counts = {
        step: len(list(steps))
        for step, steps in itertools.groupby(
            sorted(step for xs in visited_locations.values() for step in xs.values())
        )
    }
    # visit_counts_ = visit_counts.visit_counts

    # def _counts_str(c: dict[int, int]) -> Iterator[str]:
    #    for step, count in sorted(c.items()):
    #        yield f"{step}: {count}"

    # _logger.info("Visited counts: %s", ", ".join(_counts_str(visited_counts)))
    # _logger.info("Visit counts  : %s", ", ".join(_counts_str(visit_counts_)))

    return sum(
        count for step, count in visited_counts.items() if step % 2 == steps_remainder
    )
