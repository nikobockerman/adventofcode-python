from __future__ import annotations

import collections.abc
import itertools
import logging
import os
import typing
from collections import Counter
from dataclasses import InitVar, dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Sequence,
    assert_never,
    get_args,
    overload,
    override,
)

from adventofcode.tooling.directions import CardinalDirection, CardinalDirectionsAll
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
                symbol = self.get_by_xy(new_location.x, new_location.y)
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

    def _get(self, x: int, y: int) -> _MapSymbol:
        x_ = x % self.width
        y_ = y % self.height
        return super()._get(x_, y_)

    @override
    def get_by_xy(self, x: int, y: int) -> _MapSymbol:
        x_ = x % self.width
        y_ = y % self.height
        return super()._get(x_, y_)

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

        if first_corner.x > last_corner.x or first_corner.y > last_corner.y:
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
    min_steps: int
    max_steps: int

    _possible_garden_plots_at_step: int | None = field(default=None, init=False)

    def count_possible_garden_plots_at_step(self, step: int) -> int:
        if self._possible_garden_plots_at_step is not None:
            return self._possible_garden_plots_at_step

        self._possible_garden_plots_at_step = 0
        for steps in self.visits_at_step.values():
            if steps <= step and steps % 2 == step % 2:
                self._possible_garden_plots_at_step += 1
        return self._possible_garden_plots_at_step


@dataclass(slots=True, frozen=True)
class _AreaSquareResult:
    top_left_coord: Coord2d
    bottom_right_coord: Coord2d


@dataclass(slots=True, frozen=True)
class _AreaResult:
    area_square_result: _AreaSquareResult
    possible_garden_plots: int

    # Next ones are for debugging purposes
    possible_garden_plot_locations: set[Coord2d]
    locations_with_step_le_max_step: dict[Coord2d, int]

    step: InitVar[int]

    def __post_init__(self, step: int) -> None:
        locations_len = len(self.possible_garden_plot_locations)
        possible_locs = list(
            _get_possible_locations(self.locations_with_step_le_max_step.items(), step)
        )
        possible_len = len(possible_locs)
        assert locations_len == self.possible_garden_plots
        assert possible_len == self.possible_garden_plots


def _get_locations_le_step(
    locations_with_step: Iterable[tuple[Coord2d, int]], step: int
) -> Iterator[tuple[Coord2d, int]]:
    for coord, coord_step in locations_with_step:
        if coord_step <= step:
            yield coord, coord_step


def _get_possible_locations(
    locations_with_step: Iterable[tuple[Coord2d, int]], step: int
) -> Iterator[Coord2d]:
    steps_remainder = step % 2
    for coord, coord_step in _get_locations_le_step(locations_with_step, step):
        if coord_step % 2 == steps_remainder:
            yield coord


@dataclass(slots=True)
class _ExtendableDirection:
    base: _AreaSquare
    next_: _AreaSquare

    def __post_init__(self) -> None:
        max_increase = self.next_.max_steps - self.base.max_steps
        min_increase = self.next_.min_steps - self.base.min_steps
        # if self.base.top_left_corner != Coord2d(0, -11):
        assert max_increase == min_increase

        y_increase = self.next_.top_left_corner.y - self.base.top_left_corner.y
        x_increase = self.next_.top_left_corner.x - self.base.top_left_corner.x

        assert self.next_.visits_at_step.keys() == {
            Coord2d(b.x + x_increase, b.y + y_increase)
            for b in self.base.visits_at_step
        }

        step_diff_all: int | None = None
        for base_coord, base_step in self.base.visits_at_step.items():
            next_coord = Coord2d(base_coord.x + x_increase, base_coord.y + y_increase)
            next_step = self.next_.visits_at_step[next_coord]
            step_diff = next_step - base_step
            if step_diff_all is None:
                step_diff_all = step_diff
            else:
                assert step_diff_all == step_diff
        assert step_diff_all == max_increase

    def _get_full_square_area_results(
        self,
        step: int,
        does_area_need_processing: Callable[[_AreaSquareResult], bool],
    ) -> Iterator[_AreaResult]:
        for area_square in (self.base, self.next_):
            if area_square.max_steps > step:
                continue

            area = _AreaSquareResult(
                area_square.top_left_corner, area_square.bottom_right_corner
            )
            if does_area_need_processing(area):
                visits_le_step = dict(
                    _get_locations_le_step(area_square.visits_at_step.items(), step)
                )
                possible_locs = set(
                    _get_possible_locations(visits_le_step.items(), step)
                )
                yield _AreaResult(
                    area, len(possible_locs), possible_locs, visits_le_step, step
                )

        if self.next_.max_steps > step:
            return

        max_steps_increase = self.next_.max_steps - self.base.max_steps
        full_followup_areas = (step - self.next_.max_steps) // max_steps_increase

        x_increase = self.next_.top_left_corner.x - self.base.top_left_corner.x
        y_increase = self.next_.top_left_corner.y - self.base.top_left_corner.y
        step_increase = next(iter(self.next_.visits_at_step.values())) - next(
            iter(self.base.visits_at_step.values())
        )
        assert step_increase == max_steps_increase

        for i in range(1, full_followup_areas + 1):
            x_increase_area = x_increase * i
            y_increase_area = y_increase * i
            step_increase_area = step_increase * i
            area = _AreaSquareResult(
                Coord2d(
                    self.next_.top_left_corner.x + x_increase_area,
                    self.next_.top_left_corner.y + y_increase_area,
                ),
                Coord2d(
                    self.next_.bottom_right_corner.x + x_increase_area,
                    self.next_.bottom_right_corner.y + y_increase_area,
                ),
            )
            if does_area_need_processing(area):
                visits_at_step = {
                    Coord2d(
                        next_coord.x + x_increase_area,
                        next_coord.y + y_increase_area,
                    ): next_step + step_increase_area
                    for next_coord, next_step in self.next_.visits_at_step.items()
                }
                visits_le_step = dict(
                    _get_locations_le_step(visits_at_step.items(), step)
                )
                possible_locs = set(
                    _get_possible_locations(visits_le_step.items(), step)
                )
                yield _AreaResult(
                    area, len(possible_locs), possible_locs, visits_le_step, step
                )

    def _get_partial_square_area_results(
        self,
        step: int,
        does_area_need_processing: Callable[[_AreaSquareResult], bool],
        full_area_count: int,
    ) -> Iterator[_AreaResult]:
        if full_area_count == 0 and self.base.min_steps <= step:
            assert step < self.base.max_steps
            area_square_result = _AreaSquareResult(
                self.base.top_left_corner, self.base.bottom_right_corner
            )
            if does_area_need_processing(area_square_result):
                visits_le_step = dict(
                    _get_locations_le_step(self.base.visits_at_step.items(), step)
                )
                yield _AreaResult(
                    area_square_result,
                    self.base.count_possible_garden_plots_at_step(step),
                    set(_get_possible_locations(visits_le_step.items(), step)),
                    visits_le_step,
                    step,
                )

        if full_area_count == 1 and self.next_.min_steps <= step:
            assert step < self.next_.max_steps
            area_square_result = _AreaSquareResult(
                self.next_.top_left_corner, self.next_.bottom_right_corner
            )
            if does_area_need_processing(area_square_result):
                visits_le_step = dict(
                    _get_locations_le_step(self.next_.visits_at_step.items(), step)
                )
                yield _AreaResult(
                    area_square_result,
                    self.next_.count_possible_garden_plots_at_step(step),
                    set(_get_possible_locations(visits_le_step.items(), step)),
                    visits_le_step,
                    step,
                )

        full_followup_areas = full_area_count - 2
        min_steps_increase = self.next_.min_steps - self.base.min_steps

        min_last_followup_area = (
            self.next_.min_steps + full_followup_areas * min_steps_increase
        )
        # full, remainder = divmod(step - min_last_followup_area, min_steps_increase)
        # partial_followup_areas = full
        # if remainder > 0:
        #    partial_followup_areas += 1
        partial_followup_areas = (step - min_last_followup_area) // min_steps_increase
        # assert partial_followup_areas >= 0
        assert partial_followup_areas <= 2, "This might not be correct assumption"

        if partial_followup_areas <= 0:
            return

        y_increase = self.next_.top_left_corner.y - self.base.top_left_corner.y
        x_increase = self.next_.top_left_corner.x - self.base.top_left_corner.x
        step_increase = next(iter(self.next_.visits_at_step.values())) - next(
            iter(self.base.visits_at_step.values())
        )

        for i in range(1, partial_followup_areas + 1):
            x_increase_area = x_increase * (full_followup_areas + i)
            y_increase_area = y_increase * (full_followup_areas + i)
            step_increase_area = step_increase * (full_followup_areas + i)

            area = _AreaSquareResult(
                Coord2d(
                    self.next_.top_left_corner.x + x_increase_area,
                    self.next_.top_left_corner.y + y_increase_area,
                ),
                Coord2d(
                    self.next_.bottom_right_corner.x + x_increase_area,
                    self.next_.bottom_right_corner.y + y_increase_area,
                ),
            )
            if not does_area_need_processing(area):
                continue

            steps_remainder = step % 2

            counted_plots_for_partial_area = 0
            locations_le_step = dict[Coord2d, int]()
            for next_coord, next_step in self.next_.visits_at_step.items():
                followup_coord = Coord2d(
                    next_coord.x + x_increase_area,
                    next_coord.y + y_increase_area,
                )
                followup_step = next_step + step_increase_area

                if followup_step <= step:
                    if followup_step % 2 == steps_remainder:
                        counted_plots_for_partial_area += 1
                    locations_le_step[followup_coord] = followup_step

            yield _AreaResult(
                area,
                counted_plots_for_partial_area,
                set(_get_possible_locations(locations_le_step.items(), step)),
                locations_le_step,
                step,
            )

    def get_all_area_results(
        self,
        step: int,
        does_area_need_processing: Callable[[_AreaSquareResult], bool],
    ) -> Iterator[_AreaResult]:
        full_area_count = 0

        def internal_does_area_need_processing(area: _AreaSquareResult) -> bool:
            nonlocal full_area_count
            full_area_count += 1
            return does_area_need_processing(area)

        for area in self._get_full_square_area_results(
            step, internal_does_area_need_processing
        ):
            yield area

        # partial_area_count = 0
        for area in self._get_partial_square_area_results(
            step, does_area_need_processing, full_area_count
        ):
            yield area
            # partial_area_count += 1
        # assert partial_area_count <= 1


@dataclass(slots=True)
class _ExtendableArea:
    """Contains 4 quadrants.

    II  | I
    --- + --
    III | IV
    """

    quadrant1: _AreaSquare
    quadrant2: _AreaSquare
    quadrant3: _AreaSquare
    quadrant4: _AreaSquare

    base_quadrant: Literal[1, 2, 3, 4]

    def __post_init__(self) -> None:
        max_increase_area: int | None = None
        min_increase_area: int | None = None
        step_diff_area: int | None = None

        for base, next_ in (
            self._get_quadrant_pair_for_row(0),
            self._get_quadrant_pair_for_row(1),
            self._get_quadrant_pair_for_column(0),
            self._get_quadrant_pair_for_column(1),
        ):
            max_increase = next_.max_steps - base.max_steps
            min_increase = next_.min_steps - base.min_steps
            assert max_increase == min_increase

            if max_increase_area is None:
                max_increase_area = max_increase
            else:
                assert max_increase_area == max_increase
            if min_increase_area is None:
                min_increase_area = min_increase
            else:
                assert min_increase_area == min_increase

            y_increase = next_.top_left_corner.y - base.top_left_corner.y
            x_increase = next_.top_left_corner.x - base.top_left_corner.x

            assert next_.visits_at_step.keys() == {
                Coord2d(b.x + x_increase, b.y + y_increase) for b in base.visits_at_step
            }

            for base_coord, base_step in base.visits_at_step.items():
                next_coord = Coord2d(
                    base_coord.x + x_increase, base_coord.y + y_increase
                )
                next_step = next_.visits_at_step[next_coord]
                step_diff = next_step - base_step
                if step_diff_area is None:
                    step_diff_area = step_diff
                else:
                    assert step_diff_area == step_diff

    def _get_quadrant_pair_for_row(
        self, row: Literal[0, 1]
    ) -> tuple[_AreaSquare, _AreaSquare]:
        if self.base_quadrant == 1:
            if row == 0:
                return (self.quadrant1, self.quadrant2)
            if row == 1:
                return (self.quadrant4, self.quadrant3)
            assert_never(row)
        if self.base_quadrant == 2:
            if row == 0:
                return (self.quadrant2, self.quadrant1)
            if row == 1:
                return (self.quadrant3, self.quadrant4)
            assert_never(row)
        if self.base_quadrant == 3:
            if row == 0:
                return (self.quadrant3, self.quadrant4)
            if row == 1:
                return (self.quadrant2, self.quadrant1)
            assert_never(row)
        if self.base_quadrant == 4:
            if row == 0:
                return (self.quadrant4, self.quadrant3)
            if row == 1:
                return (self.quadrant1, self.quadrant2)
            assert_never(row)
        assert_never(self.base_quadrant)

    def _get_quadrant_pair_for_column(
        self, column: Literal[0, 1]
    ) -> tuple[_AreaSquare, _AreaSquare]:
        if self.base_quadrant == 1:
            if column == 0:
                return (self.quadrant1, self.quadrant4)
            if column == 1:
                return (self.quadrant2, self.quadrant3)
            assert_never(column)
        if self.base_quadrant == 2:
            if column == 0:
                return (self.quadrant2, self.quadrant3)
            if column == 1:
                return (self.quadrant1, self.quadrant4)
            assert_never(column)
        if self.base_quadrant == 3:
            if column == 0:
                return (self.quadrant3, self.quadrant2)
            if column == 1:
                return (self.quadrant4, self.quadrant1)
            assert_never(column)
        if self.base_quadrant == 4:
            if column == 0:
                return (self.quadrant4, self.quadrant1)
            if column == 1:
                return (self.quadrant3, self.quadrant2)
            assert_never(column)
        assert_never(self.base_quadrant)

    def _max_full_followup_areas_for_row(
        self, max_steps: int, *, row: Literal[0, 1] = 0
    ) -> int:
        q_base, q_next = self._get_quadrant_pair_for_row(row)
        max_steps_increase = q_next.max_steps - q_base.max_steps
        return (max_steps - q_next.max_steps) // max_steps_increase

    def get_all_extendable_directions(
        self, max_steps: int
    ) -> Iterator[_ExtendableDirection]:
        pair_row_0 = self._get_quadrant_pair_for_row(0)
        pair_row_1 = self._get_quadrant_pair_for_row(1)
        for base, next_ in (
            pair_row_0,
            pair_row_1,
            self._get_quadrant_pair_for_column(0),
            self._get_quadrant_pair_for_column(1),
        ):
            if base.min_steps > max_steps:
                continue
            yield _ExtendableDirection(base, next_)

        max_full_followup_areas = self._max_full_followup_areas_for_row(max_steps)
        if max_full_followup_areas <= 0:
            return

        max_full_followup_areas_for_next_row = self._max_full_followup_areas_for_row(
            max_steps, row=1
        )

        assert max_full_followup_areas >= max_full_followup_areas_for_next_row
        assert max_full_followup_areas - 1 <= max_full_followup_areas_for_next_row

        step_increase = next(iter(pair_row_0[1].visits_at_step.values())) - next(
            iter(pair_row_0[0].visits_at_step.values())
        )
        x_increase = pair_row_0[1].top_left_corner.x - pair_row_0[0].top_left_corner.x
        y_increase = pair_row_0[1].top_left_corner.y - pair_row_0[0].top_left_corner.y
        min_increase = pair_row_0[1].min_steps - pair_row_0[0].min_steps
        max_increase = pair_row_0[1].max_steps - pair_row_0[0].max_steps

        prev_row_0 = pair_row_0[1]
        prev_row_1 = pair_row_1[1]

        def create_next_area_square(prev: _AreaSquare) -> _AreaSquare:
            return _AreaSquare(
                Coord2d(
                    prev.top_left_corner.x + x_increase,
                    prev.top_left_corner.y + y_increase,
                ),
                Coord2d(
                    prev.bottom_right_corner.x + x_increase,
                    prev.bottom_right_corner.y + y_increase,
                ),
                visits_at_step={
                    Coord2d(
                        prev_coord.x + x_increase, prev_coord.y + y_increase
                    ): prev_step + step_increase
                    for prev_coord, prev_step in prev.visits_at_step.items()
                },
                min_steps=prev.min_steps + min_increase,
                max_steps=prev.max_steps + max_increase,
            )

        for _ in range(max_full_followup_areas):
            next_row_0 = create_next_area_square(prev_row_0)
            next_row_1 = create_next_area_square(prev_row_1)

            yield _ExtendableDirection(next_row_0, next_row_1)

            prev_row_0 = next_row_0
            prev_row_1 = next_row_1


@dataclass(slots=True, kw_only=True)
class _AreasAroundStart:
    top_left: _ExtendableArea
    top_right: _ExtendableArea
    bottom_left: _ExtendableArea
    bottom_right: _ExtendableArea
    center_to_top: _ExtendableDirection
    center_to_right: _ExtendableDirection
    center_to_bottom: _ExtendableDirection
    center_to_left: _ExtendableDirection
    completed_non_extendable_areas: list[_AreaSquare] = field(default_factory=list)

    def all_areas(self) -> Iterator[_AreaSquare]:
        for area in (
            self.top_left,
            self.top_right,
            self.bottom_left,
            self.bottom_right,
        ):
            yield area.quadrant1
            yield area.quadrant2
            yield area.quadrant3
            yield area.quadrant4

        for direction in (
            self.center_to_top,
            self.center_to_right,
            self.center_to_bottom,
            self.center_to_left,
        ):
            yield direction.base
            yield direction.next_

        yield from self.completed_non_extendable_areas

    def get_all_extendable_directions(
        self, max_steps: int
    ) -> Iterator[_ExtendableDirection]:
        yield self.center_to_top
        for area in (
            self.top_left,
            self.top_right,
            self.bottom_left,
            self.bottom_right,
        ):
            yield from area.get_all_extendable_directions(max_steps)
        yield self.center_to_right
        yield self.center_to_bottom
        yield self.center_to_left

    def get_non_extendable_area_results(self, step: int) -> Iterator[_AreaResult]:
        for area in self.completed_non_extendable_areas:
            visits_le_step = dict(
                _get_locations_le_step(area.visits_at_step.items(), step)
            )
            yield _AreaResult(
                _AreaSquareResult(area.top_left_corner, area.bottom_right_corner),
                area.count_possible_garden_plots_at_step(step),
                set(_get_possible_locations(visits_le_step.items(), step)),
                visits_le_step,
                step,
            )


_map: _InfiniteMap | None = None


def _map_of_visited_area(area: Any) -> Iterator[str]:
    assert _map is not None
    min_x = area.top_left.x
    min_y = area.top_left.y
    max_x = area.bottom_right.x
    max_y = area.bottom_right.y
    l = len(str(max(area.visits_at_step.values())))
    yield f"Steps for area x:{min_x:d}-{max_x:d}, y={min_y:d}-{max_y:d}"
    yield "-" * ((l + 1) * (max_x - min_x + 1))
    for y in range(min_y, max_y + 1):
        row = ""
        for x in range(min_x, max_x + 1):
            coord = Coord2d(x, y)
            if coord == _map.start:
                row += " " * l + "S"
            elif coord in area.visits_at_step:
                row += f" {area.visits_at_step[coord]:{l}d}"
            else:
                row += " " * l + "#"
        yield row


def _map_locations(visited: set[Coord2d]) -> Iterator[str]:
    assert _map is not None
    min_x = min(x.x for x in visited)
    min_y = min(x.y for x in visited)
    max_x = max(x.x for x in visited)
    max_y = max(x.y for x in visited)
    yield (
        f"Count: {len(visited)}. "
        f"Area: x={min_x:d} - {max_x:d}, y={min_y:d} - {max_y:d}"
    )
    yield "-" * (max_x - min_x + 1)
    for y in range(min_y, max_y + 1):
        row = ""
        for x in range(min_x, max_x + 1):
            coord = Coord2d(x, y)
            if coord == _map.start:
                v = "S"
            elif coord in visited:
                v = "O"
            else:
                v = _map.get_by_xy(x, y)
            row += f"{v}"
        yield row


def _get_area_min_max(coords: Iterable[Coord2d]) -> tuple[int, int, int, int]:
    min_x: int | None = None
    max_x: int | None = None
    min_y: int | None = None
    max_y: int | None = None
    for coord in coords:
        if min_x is None or coord.x < min_x:
            min_x = coord.x
        if max_x is None or coord.x > max_x:
            max_x = coord.x
        if min_y is None or coord.y < min_y:
            min_y = coord.y
        if max_y is None or coord.y > max_y:
            max_y = coord.y
    assert min_x is not None
    assert max_x is not None
    assert min_y is not None
    assert max_y is not None
    return min_x, min_y, max_x, max_y


def _map_overlay_visited_with_required(
    visited: set[Coord2d], required: set[Coord2d]
) -> Iterator[str]:
    assert _map is not None
    min_x, min_y, max_x, max_y = _get_area_min_max(itertools.chain(visited, required))
    yield (
        f"Visited: {len(visited)}. "
        f"Required: {len(required)}. "
        f"Area: x={min_x:d} - {max_x:d}, y={min_y:d} - {max_y:d}"
    )
    yield "-" * (max_x - min_x + 1)
    for y in range(min_y, max_y + 1):
        row = ""
        for x in range(min_x, max_x + 1):
            coord = Coord2d(x, y)
            if coord == _map.start:
                v = "S"
            elif coord in visited:
                v = "o" if coord in required else "v"
            elif coord in required:
                v = "X"
            else:
                v = _map.get_by_xy(x, y)
                if v == "#":
                    v = "_"
            row += f"{v}"
        yield row


def _p2_step1_resolve_around_start(
    map_: _InfiniteMap,
) -> tuple[_AreasAroundStart, dict[Coord2d, int]]:
    @dataclass(slots=True)
    class _RangeArea:
        top_left: Coord2d
        bottom_right: Coord2d
        visits_at_step: dict[Coord2d, int] = field(default_factory=dict, init=False)
        _min_step: int | None = field(default=None, init=False)
        _max_step: int | None = field(default=None, init=False)

        @property
        def min_step(self) -> int:
            assert self._min_step is not None
            return self._min_step

        @property
        def max_step(self) -> int:
            assert self._max_step is not None
            return self._max_step

        def width(self) -> int:
            return self.bottom_right.x - self.top_left.x + 1

        def height(self) -> int:
            return self.bottom_right.y - self.top_left.y + 1

        def add_visits(self, step: int, visits: set[Coord2d]) -> None:
            for coord in visits:
                if not self.contains(coord):
                    continue

                self.visits_at_step[coord] = step
                if self._min_step is None or step < self._min_step:
                    self._min_step = step
                if self._max_step is None or step > self._max_step:
                    self._max_step = step

        def contains(self, coord: Coord2d) -> bool:
            return (
                self.top_left.x <= coord.x <= self.bottom_right.x
                and self.top_left.y <= coord.y <= self.bottom_right.y
            )

        def to_square(self) -> _AreaSquare:
            return _AreaSquare(
                self.top_left,
                self.bottom_right,
                self.visits_at_step,
                self.min_step,
                self.max_step,
            )

    class _RangeAreaQuadrants:
        def __init__(
            self, map_: _InfiniteMap, quadrant_from_center: Literal[1, 2, 3, 4]
        ) -> None:
            self.base_quadrant: Literal[1, 2, 3, 4]
            if quadrant_from_center == 1:
                top_left_coord = Coord2d(
                    map_.last_x + 1, map_.first_y - 2 * map_.height
                )
                self.base_quadrant = 3
            elif quadrant_from_center == 2:
                top_left_coord = Coord2d(
                    map_.first_x - 2 * map_.width, map_.first_y - 2 * map_.height
                )
                self.base_quadrant = 4
            elif quadrant_from_center == 3:
                top_left_coord = Coord2d(map_.first_x - 2 * map_.width, map_.last_y + 1)
                self.base_quadrant = 1
            elif quadrant_from_center == 4:
                top_left_coord = Coord2d(map_.last_x + 1, map_.last_y + 1)
                self.base_quadrant = 2
            else:
                assert_never(quadrant_from_center)

            self.q2 = _RangeArea(
                top_left_coord,
                Coord2d(
                    top_left_coord.x + map_.width - 1,
                    top_left_coord.y + map_.height - 1,
                ),
            )
            self.q1 = _RangeArea(
                Coord2d(self.q2.top_left.x + map_.width, self.q2.top_left.y),
                Coord2d(self.q2.bottom_right.x + map_.width, self.q2.bottom_right.y),
            )
            self.q3 = _RangeArea(
                Coord2d(self.q2.top_left.x, self.q2.top_left.y + map_.height),
                Coord2d(self.q2.bottom_right.x, self.q2.bottom_right.y + map_.height),
            )
            self.q4 = _RangeArea(
                Coord2d(self.q3.top_left.x + map_.width, self.q3.top_left.y),
                Coord2d(self.q3.bottom_right.x + map_.width, self.q3.bottom_right.y),
            )

        def to_result(self) -> _ExtendableArea:
            return _ExtendableArea(
                self.q1.to_square(),
                self.q2.to_square(),
                self.q3.to_square(),
                self.q4.to_square(),
                self.base_quadrant,
            )

        def all_ranges(self) -> Iterator[_RangeArea]:
            yield self.q1
            yield self.q2
            yield self.q3
            yield self.q4

    class _RangeAreaVector:
        @overload
        def __init__(
            self,
            map_: _InfiniteMap,
            center: _RangeArea,
            direction_from_center: CardinalDirection,
            /,
        ) -> None:
            pass

        @overload
        def __init__(self, base: _RangeArea, next_: _RangeArea, /) -> None:
            pass

        def __init__(
            self,
            map_or_base: _InfiniteMap | _RangeArea,
            next_or_center: _RangeArea,
            direction_from_center: CardinalDirection | None = None,
        ) -> None:
            if isinstance(map_or_base, _RangeArea):
                self.base = map_or_base
                self.next = next_or_center
                return

            map_ = map_or_base
            center = next_or_center
            assert direction_from_center is not None
            if direction_from_center == CardinalDirection.N:
                self.base = _RangeArea(
                    Coord2d(center.top_left.x, center.top_left.y - map_.height),
                    Coord2d(center.bottom_right.x, center.top_left.y - 1),
                )
                self.next = _RangeArea(
                    Coord2d(self.base.top_left.x, self.base.top_left.y - map_.height),
                    Coord2d(self.base.bottom_right.x, self.base.top_left.y - 1),
                )
            elif direction_from_center == CardinalDirection.S:
                self.base = _RangeArea(
                    Coord2d(center.top_left.x, center.bottom_right.y + 1),
                    Coord2d(center.bottom_right.x, center.bottom_right.y + map_.height),
                )
                self.next = _RangeArea(
                    Coord2d(self.base.top_left.x, self.base.bottom_right.y + 1),
                    Coord2d(
                        self.base.bottom_right.x, self.base.bottom_right.y + map_.height
                    ),
                )
            elif direction_from_center == CardinalDirection.E:
                self.base = _RangeArea(
                    Coord2d(center.bottom_right.x + 1, center.top_left.y),
                    Coord2d(center.bottom_right.x + map_.width, center.bottom_right.y),
                )
                self.next = _RangeArea(
                    Coord2d(self.base.bottom_right.x + 1, self.base.top_left.y),
                    Coord2d(
                        self.base.bottom_right.x + map_.width, self.base.bottom_right.y
                    ),
                )
            elif direction_from_center == CardinalDirection.W:
                self.base = _RangeArea(
                    Coord2d(center.top_left.x - map_.width, center.top_left.y),
                    Coord2d(center.top_left.x - 1, center.bottom_right.y),
                )
                self.next = _RangeArea(
                    Coord2d(self.base.top_left.x - map_.width, self.base.top_left.y),
                    Coord2d(self.base.top_left.x - 1, self.base.bottom_right.y),
                )
            else:
                assert_never(direction_from_center)

        def get_next(self) -> _RangeAreaVector:
            x_increase = self.next.top_left.x - self.base.top_left.x
            y_increase = self.next.top_left.y - self.base.top_left.y
            return _RangeAreaVector(
                self.next,
                _RangeArea(
                    Coord2d(
                        self.next.top_left.x + x_increase,
                        self.next.top_left.y + y_increase,
                    ),
                    Coord2d(
                        self.next.bottom_right.x + x_increase,
                        self.next.bottom_right.y + y_increase,
                    ),
                ),
            )

        def is_extendable(self) -> bool:
            max_diff = self.next.max_step - self.base.max_step
            min_diff = self.next.min_step - self.base.min_step
            if max_diff != min_diff:
                # _logger.debug(
                #    "\n%s",
                #    "\n".join(
                #        itertools.chain(
                #            _map_of_visited_area(self.base),
                #            _map_of_visited_area(self.next),
                #        )
                #    ),
                # )
                return False

            y_increase = self.next.top_left.y - self.base.top_left.y
            x_increase = self.next.top_left.x - self.base.top_left.x

            prev_step_diff: int | None = None
            for base_coord, base_step in self.base.visits_at_step.items():
                next_coord = Coord2d(
                    base_coord.x + x_increase, base_coord.y + y_increase
                )
                next_step = self.next.visits_at_step[next_coord]
                step_diff = next_step - base_step

                if prev_step_diff is None:
                    prev_step_diff = step_diff
                elif prev_step_diff != step_diff:
                    # _logger.debug(
                    #    "\n%s",
                    #    "\n".join(
                    #        itertools.chain(
                    #            _map_of_visited_area(self.base),
                    #            _map_of_visited_area(self.next),
                    #        )
                    #    ),
                    # )
                    return False

            return True

        def to_result(self) -> _ExtendableDirection:
            return _ExtendableDirection(
                base=self.base.to_square(), next_=self.next.to_square()
            )

        def all_areas(self) -> Iterator[_RangeArea]:
            yield self.base
            yield self.next

    class _RangeAreasAroundStart:
        def __init__(self, map_: _InfiniteMap) -> None:
            self.top_left = _RangeAreaQuadrants(map_, 2)
            self.top_right = _RangeAreaQuadrants(map_, 1)
            self.bottom_left = _RangeAreaQuadrants(map_, 3)
            self.bottom_right = _RangeAreaQuadrants(map_, 4)
            center = _RangeArea(
                Coord2d(map_.first_x, map_.first_y), Coord2d(map_.last_x, map_.last_y)
            )
            self.center_to_top = _RangeAreaVector(map_, center, CardinalDirection.N)
            self.center_to_right = _RangeAreaVector(map_, center, CardinalDirection.E)
            self.center_to_bottom = _RangeAreaVector(map_, center, CardinalDirection.S)
            self.center_to_left = _RangeAreaVector(map_, center, CardinalDirection.W)
            self.completed_non_extendable_areas = list((center,))

        def add_visits(self, step: int, visits: set[Coord2d]) -> None:
            for area in self.all_areas():
                area.add_visits(step, visits)

        def to_result(self) -> _AreasAroundStart:
            return _AreasAroundStart(
                top_left=self.top_left.to_result(),
                top_right=self.top_right.to_result(),
                bottom_left=self.bottom_left.to_result(),
                bottom_right=self.bottom_right.to_result(),
                center_to_top=self.center_to_top.to_result(),
                center_to_right=self.center_to_right.to_result(),
                center_to_bottom=self.center_to_bottom.to_result(),
                center_to_left=self.center_to_left.to_result(),
                completed_non_extendable_areas=list(
                    area.to_square() for area in self.completed_non_extendable_areas
                ),
            )

        def all_areas(self) -> Iterator[_RangeArea]:
            yield from self.top_left.all_ranges()
            yield from self.top_right.all_ranges()
            yield from self.bottom_left.all_ranges()
            yield from self.bottom_right.all_ranges()

            for vector in self.all_vectors():
                yield from vector.all_areas()

            yield from self.completed_non_extendable_areas

        def all_vectors(self) -> Iterator[_RangeAreaVector]:
            yield self.center_to_top
            yield self.center_to_right
            yield self.center_to_bottom
            yield self.center_to_left

        def check_complete(self) -> list[_RangeArea]:
            new_areas: list[_RangeArea] = []
            if not self.center_to_top.is_extendable():
                self.completed_non_extendable_areas.append(self.center_to_top.base)
                self.center_to_top = self.center_to_top.get_next()
                new_areas.append(self.center_to_top.next)
            if not self.center_to_right.is_extendable():
                self.completed_non_extendable_areas.append(self.center_to_right.base)
                self.center_to_right = self.center_to_right.get_next()
                new_areas.append(self.center_to_right.next)
            if not self.center_to_bottom.is_extendable():
                self.completed_non_extendable_areas.append(self.center_to_bottom.base)
                self.center_to_bottom = self.center_to_bottom.get_next()
                new_areas.append(self.center_to_bottom.next)
            if not self.center_to_left.is_extendable():
                self.completed_non_extendable_areas.append(self.center_to_left.base)
                self.center_to_left = self.center_to_left.get_next()
                new_areas.append(self.center_to_left.next)
            return new_areas

    range_areas = _RangeAreasAroundStart(map_)

    for area in range_areas.all_areas():
        assert map_.width == area.width()
        assert map_.height == area.height()

    required_visited_locations = set[Coord2d](
        Coord2d(x, y)
        for area in range_areas.all_areas()
        for y, xs in map_.iter_data(area.top_left, area.bottom_right)
        for x, symbol in xs
        if symbol == "."
        and not all(
            map_.get_by_xy(adjoin.x, adjoin.y) == "#"
            for _, adjoin in Coord2d(x, y).adjoins()
        )
    )

    def get_visited_locations_at_step_with_duplicate_check(
        visits_at_step: dict[int, set[Coord2d]],
    ) -> dict[Coord2d, int]:
        result: dict[Coord2d, int] = {}
        for step, locations in visits_at_step.items():
            for location in locations:
                assert location not in result, "Duplicate step for location"
                result[location] = step
        return result

    visited_locations_at_step: dict[int, set[Coord2d]] = {0: {map_.start}}
    visited_locations: set[Coord2d] = {map_.start}
    locations_to_process: set[Coord2d] = {map_.start}
    remaining_locations_to_visit = set(required_visited_locations) - visited_locations
    range_areas.add_visits(0, visited_locations)
    step = 0
    min_step: int | None = None
    cont = True
    while cont:
        _logger.debug(
            "Required distance: %s",
            max(map_.start.distance_to_cardinal(x) for x in required_visited_locations),
        )
        _logger.debug(
            "Required locations: \n%s",
            "\n".join(_map_locations(required_visited_locations)),
        )

        # while not required_visited_locations <= visited_locations or (
        #    min_step is not None and step < min_step
        # ):
        while remaining_locations_to_visit:
            step += 1
            # with open(f"visited-with-required-map-{step}.txt", "w") as f:
            #    f.write(
            #        "\n".join(
            #            _map_overlay_visited_with_required(
            #                visited_locations, required_visited_locations
            #            )
            #        )
            #    )
            # _logger.debug(
            #    "Visited with required: \n%s",
            #    "\n".join(
            #        _map_overlay_visited_with_required(
            #            visited_locations, required_visited_locations
            #        )
            #    ),
            # )
            _logger.debug("Step: %s", step)
            adjoin_locations = set(
                map_.get_adjoin_garden_plots_for_locations(locations_to_process)
            )
            adjoin_locations -= visited_locations
            range_areas.add_visits(step, adjoin_locations)
            visited_locations_at_step[step] = adjoin_locations
            visited_locations.update(adjoin_locations)
            remaining_locations_to_visit -= adjoin_locations
            locations_to_process = adjoin_locations
            # _logger.debug("Visited: \n%s", "\n".join(_map_visited_str(visited_locations)))
        # _logger.debug(
        #    "Initial visited: \n%s",
        #    "\n".join(
        #        _map_step_visited_str(
        #            get_visited_locations_at_step_with_duplicate_check(
        #                visited_locations_at_step
        #            )
        #        )
        #    ),
        # )
        new_areas = range_areas.check_complete()
        if not new_areas:
            cont = False
            continue

        _logger.debug("New areas: %s", len(new_areas))
        for step, locations in visited_locations_at_step.items():
            for area in new_areas:
                area.add_visits(step, locations)

        required_visited_locations |= set[Coord2d](
            Coord2d(x, y)
            for area in new_areas
            for y, xs in map_.iter_data(area.top_left, area.bottom_right)
            for x, symbol in xs
            if symbol == "."
            and not all(
                map_.get_by_xy(adjoin.x, adjoin.y) == "#"
                for _, adjoin in Coord2d(x, y).adjoins()
            )
        )
        remaining_locations_to_visit = required_visited_locations - visited_locations

    all_visited_locations_with_step: dict[Coord2d, int] = (
        get_visited_locations_at_step_with_duplicate_check(visited_locations_at_step)
    )

    return range_areas.to_result(), all_visited_locations_with_step


_l: int = 3
_min_x: int | None = None
_min_y: int | None = None
_max_x: int | None = None
_max_y: int | None = None


def _map_step_visited_str(visited: dict[Coord2d, int]) -> Iterator[str]:
    assert _map is not None
    global _min_x
    global _min_y
    global _max_x
    global _max_y
    min_x = min(x.x for x in visited)
    max_x = max(x.x for x in visited)
    min_y = min(x.y for x in visited)
    max_y = max(x.y for x in visited)
    _min_x = min_x
    _min_y = min_y
    _max_x = max_x
    _max_y = max_y
    assert len(str(max(visited.values()))) + 1 == _l
    yield f"Steps for area x={min_x:d} - {max_x:d}, y={min_y:d} - {max_y:d}"
    yield "-" * (_l * (max_x - min_x + 1))
    for y in range(min_y, max_y + 1):
        row = ""
        for x in range(min_x, max_x + 1):
            coord = Coord2d(x, y)
            if coord == _map.start:
                v = "S"
            elif coord in visited:
                v = visited[coord]
            else:
                v = _map.get_by_xy(x, y)
            row += f"{v:>{_l}}"
        yield row


def _map_of_step_counts(areas: Iterable[_AreaResult]) -> Iterator[str]:
    assert _map is not None
    assert _min_x is not None
    assert _min_y is not None
    assert _max_x is not None
    assert _max_y is not None
    min_x = _min_x
    min_y = _min_y
    max_x = _max_x
    max_y = _max_y
    visited_locations_with_step: dict[Coord2d, int] = {}
    for area in areas:
        for location, step in area.locations_with_step_le_max_step.items():
            assert (
                location not in visited_locations_with_step
            ), "Duplicate step for location"
            visited_locations_with_step[location] = step
    yield (
        f"Step counts at requested step. Count: {len(visited_locations_with_step)}. "
        f"Full area: x={min_x:d} - {max_x:d}, y={min_y:d} - {max_y:d}"
    )
    yield "-" * (_l * (max_x - min_x + 1))
    for y in range(min_y, max_y + 1):
        row = ""
        for x in range(min_x, max_x + 1):
            coord = Coord2d(x, y)
            if coord == _map.start:
                v = "S"
            elif coord in visited_locations_with_step:
                v = visited_locations_with_step[coord]
            else:
                v = _map.get_by_xy(x, y)
            row += f"{v:>{_l}}"
        yield row


def _map_of_result(areas: Iterable[_AreaResult]) -> Iterator[str]:
    assert _map is not None
    assert _min_x is not None
    assert _min_y is not None
    assert _max_x is not None
    assert _max_y is not None
    min_x = _min_x
    min_y = _min_y
    max_x = _max_x
    max_y = _max_y
    areas = list(areas)
    visited_locations: set[Coord2d] = set(
        itertools.chain(*(a.possible_garden_plot_locations for a in areas))
    )
    yield (
        f"Locations at requested step. Count: {len(visited_locations)}. "
        f"Full area: x={min_x:d} - {max_x:d}, y={min_y:d} - {max_y:d}"
    )
    yield "-" * (_l * (max_x - min_x + 1))
    for y in range(min_y, max_y + 1):
        row = ""
        for x in range(min_x, max_x + 1):
            coord = Coord2d(x, y)
            if coord == _map.start:
                v = "+" if coord in visited_locations else "S"
            elif coord in visited_locations:
                v = "O"
            else:
                v = _map.get_by_xy(x, y)
            row += f"{v:>{_l}}"
        yield row


def _p2_step2_extend_areas(
    areas: _AreasAroundStart, step: int, initial_visits: dict[Coord2d, int]
) -> int:
    # 2. Take the corner squares and extend them outbound from stars adjusting the step
    #    counts

    # Extend top-left towards top once (TODO: Keep extending)
    processed_areas: set[_AreaSquareResult] = set()

    result_areas: list[_AreaResult] = []

    def add_result_area(area: _AreaResult) -> None:
        nonlocal result_areas
        if area.possible_garden_plots > 0:
            result_areas.append(area)

    for area in areas.get_non_extendable_area_results(step):
        add_result_area(area)

    garden_plot_count = sum(
        area.possible_garden_plots
        for area in areas.get_non_extendable_area_results(step)
    )

    # garden_plot_count = sum(
    #    area.count_possible_garden_plots_at_step(map_, step)
    #    for area in areas.completed_non_extendable_areas
    # )
    for extendable_direction in areas.get_all_extendable_directions(step):
        for area_result in extendable_direction.get_all_area_results(
            step, lambda area: area not in processed_areas
        ):
            processed_areas.add(area_result.area_square_result)
            garden_plot_count += area_result.possible_garden_plots
            add_result_area(area_result)

    # with open("correct-step-counts.txt", "w") as f:
    #    f.write("\n".join(_map_step_visited_str(initial_visits)))
    # with open("result-step-counts.txt", "w") as f:
    #    f.write("\n".join(_map_of_step_counts(result_areas)))
    # with open("result-locations.txt", "w") as f:
    #    f.write("\n".join(_map_of_result(result_areas)))
    # _logger.debug("\n%s", "\n".join(_map_of_step_counts(result_areas)))
    # _logger.debug("\n%s", "\n".join(_map_of_result(result_areas)))

    initial_locs_with_steps_le = dict(
        _get_locations_le_step(initial_visits.items(), step)
    )
    initial_locs_possible = set(
        _get_possible_locations(initial_locs_with_steps_le.items(), step)
    )
    initial_result = len(initial_locs_possible)

    locs_with_steps_le = dict(
        itertools.chain(
            *(a.locations_with_step_le_max_step.items() for a in result_areas)
        )
    )
    locs_possible = set(_get_possible_locations(locs_with_steps_le.items(), step))
    locs_result = len(locs_possible)
    new_locs_possible = locs_possible - initial_locs_possible
    missing_locs_possible = initial_locs_possible - locs_possible

    _logger.info("Calculated result: %s", garden_plot_count)
    _logger.info("Locs result: %s", locs_result)
    if max(initial_locs_with_steps_le.values()) >= step:
        _logger.info("Initial result: %s", initial_result)
        _logger.info(
            "New locs compared to initial: %s",
            "\n".join(
                str(x) for x in (sorted(new_locs_possible, key=lambda x: (x.x, x.y)))
            ),
        )
        _logger.info(
            "Missing locs compared to initial: %s",
            "\n".join(
                str(x)
                for x in (sorted(missing_locs_possible, key=lambda x: (x.x, x.y)))
            ),
        )

    return garden_plot_count


# TODO
# 1. Process width-first from start until 4x4 squares with start in the middle have all
#    garden plots visited
# 2. Take the corner squares and extend them outbound from stars adjusting the step
#    counts
# 3. Keep extending until max step count in one extended square reaches steps, once that
#    happens ignore such square
# 4. Process the remaining outer bounds with the width-first algorighm
#    (or complex if too slow)

REAL_STEPS = 26_501_365


def p2(input_str: str, steps: int = 2000) -> int:
    map_ = _InfiniteMap(input_str.splitlines())
    global _map
    _map = map_
    _logger.debug("Map size: w=%s, h=%s", map_.width, map_.height)

    # Step 1: START
    areas_around_start, all_visits = _p2_step1_resolve_around_start(map_)

    # Step 1: END

    # Step 2: START
    result = _p2_step2_extend_areas(areas_around_start, steps, all_visits)
    return result
    # Step 2: END

    coords_and_steps = [
        (coord, step)
        for area in areas_around_start.all_areas()
        for coord, step in area.visits_at_step.items()
    ]

    coords_with_steps = dict[Coord2d, set[int]]()
    for coord, step in coords_and_steps:
        steps_ = coords_with_steps.setdefault(coord, set())
        steps_.add(step)

    # Verify that there is only one step count per coord
    for coord_, steps_ in coords_with_steps.items():
        if len(steps_) > 1:
            raise AssertionError(f"{coord_} found in more than one step: {steps_}")  # noqa: TRY003

    coords_with_step: dict[Coord2d, int] = {
        coord: next(iter(steps_)) for coord, steps_ in coords_with_steps.items()
    }

    # Calculate result
    steps_remainder = steps % 2
    return sum(
        1
        for step_ in coords_with_step.values()
        if step_ % 2 == steps_remainder and step_ <= steps
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


if __name__ == "__main__":
    import pathlib

    import pyinstrument

    input_str = (pathlib.Path(__file__).parent / "input-d21.txt").read_text().strip()
    with pyinstrument.profile():
        result = p2(input_str)
    print(result)
