from __future__ import annotations

import itertools
import logging
from collections import Counter
from collections.abc import Collection
from dataclasses import InitVar, dataclass, field
from typing import (
    TYPE_CHECKING,
    Literal,
    Self,
    assert_never,
    get_args,
    overload,
    override,
)

from adventofcode.tooling.directions import CardinalDirection, CardinalDirectionsAll
from adventofcode.tooling.map import Coord2d, Map2d

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

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

        if (max_x - min_x + 1) <= self.width and start_x <= stop_x:
            yield from self._sequence_data[y_][start_x : stop_x + 1]
            return

        yield from self._sequence_data[y_][start_x:]
        while min_x < max_x - self.width:
            yield from self._sequence_data[y_]
            min_x += self.width
        yield from self._sequence_data[y_][: stop_x + 1]

    def get_row(self, y: int) -> tuple[_MapSymbol, ...]:
        data_y = y % self.height
        return self._sequence_data[data_y]

    @override
    def iter_data(
        self,
        first_corner: Coord2d | None = None,
        last_corner: Coord2d | None = None,
        *,
        columns_first: bool = False,
    ) -> Iterable[tuple[int, Iterable[tuple[int, _MapSymbol]]]]:
        if columns_first is True:
            raise NotImplementedError

        if first_corner is None or last_corner is None:
            raise NotImplementedError

        if first_corner.x > last_corner.x or first_corner.y > last_corner.y:
            raise NotImplementedError

        for y in range(first_corner.y, last_corner.y + 1):
            yield (
                y,
                zip(
                    range(first_corner.x, last_corner.x + 1),
                    self._iter_row(y, first_corner.x, last_corner.x),
                    strict=True,
                ),
            )


class _MapRow:
    def __init__(self, data: tuple[_MapSymbol, ...]) -> None:
        self.data = data
        self._len = len(data)

    def __getitem__(self, x: int) -> _MapSymbol:
        data_x = x % self._len
        return self.data[data_x]

    def __repr__(self) -> str:
        return f"MapRow({self.data!r})"


@dataclass(slots=True)
class _AreaSquareBasic:
    min_steps: int
    max_steps: int


@dataclass(slots=True)
class _AreaSquareFull(_AreaSquareBasic):
    puzzle_max_steps: InitVar[int]
    visits_at_step: InitVar[Counter[int]]

    _puzzle_max_steps: int = field(init=False)
    _possible_garden_plots_full: int = field(init=False)
    _visits_at_step: Counter[int] = field(init=False)
    _garden_plots_cache: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(
        self, puzzle_max_steps: int, visits_at_step: Counter[int]
    ) -> None:
        self._puzzle_max_steps = puzzle_max_steps
        self._visits_at_step = visits_at_step
        self._possible_garden_plots_full = self._count_possible_garden_plots_at_step(
            puzzle_max_steps
        )

    def count_possible_garden_plots_at_step(self, step: int) -> int:
        if step == self._puzzle_max_steps:
            return self._possible_garden_plots_full
        if step > self.max_steps:
            rem_correct = self._puzzle_max_steps % 2
            rem_step = step % 2
            if rem_step != rem_correct:
                return self._count_cached_possible_garden_plots_at_step(
                    self._puzzle_max_steps - 1
                )
            return self._possible_garden_plots_full
        if step < self.min_steps:
            return 0

        return self._count_cached_possible_garden_plots_at_step(step)

    def _count_cached_possible_garden_plots_at_step(self, step: int) -> int:
        if step in self._garden_plots_cache:
            return self._garden_plots_cache[step]

        possible_count = self._count_possible_garden_plots_at_step(step)
        self._garden_plots_cache[step] = possible_count
        return possible_count

    def _count_possible_garden_plots_at_step(self, step: int) -> int:
        possible_count = 0
        for steps, count in self._visits_at_step.items():
            if steps <= step and steps % 2 == step % 2:
                possible_count += count
        return possible_count


@dataclass(slots=True)
class _AreaSquareExtended(_AreaSquareBasic):
    full_origin: InitVar[_AreaSquareFull]

    _full_origin: _AreaSquareFull = field(init=False)
    _steps_offset: int = field(init=False)

    def __post_init__(self, full_origin: _AreaSquareFull) -> None:
        self._full_origin = full_origin
        self._steps_offset = self.min_steps - self._full_origin.min_steps

    def count_possible_garden_plots_at_step(self, step: int) -> int:
        return self._full_origin.count_possible_garden_plots_at_step(
            step - self._steps_offset
        )


@dataclass(slots=True, frozen=True)
class _AreaResult:
    possible_garden_plots: int


@dataclass(slots=True)
class _ExtendableDirection:
    base: _AreaSquareFull | _AreaSquareExtended
    next_: _AreaSquareFull | _AreaSquareExtended

    include_base_and_next: bool = field(default=True, kw_only=True)

    def _get_full_square_area_results(self, step: int) -> tuple[int, int]:
        result = 0
        full_area_count = 0

        if self.base.max_steps <= step:
            full_area_count += 1
            if self.include_base_and_next:
                result += self.base.count_possible_garden_plots_at_step(step)

        if self.next_.max_steps <= step:
            full_area_count += 1
            if self.include_base_and_next:
                result += self.next_.count_possible_garden_plots_at_step(step)
        else:
            return result, full_area_count

        max_steps_increase = self.next_.max_steps - self.base.max_steps
        full_followup_areas = (step - self.next_.max_steps) // max_steps_increase

        if full_followup_areas == 0:
            return result, full_area_count

        full_area_count += full_followup_areas
        next_counts = full_followup_areas // 2
        base_counts = full_followup_areas - next_counts
        result += base_counts * self.base.count_possible_garden_plots_at_step(step)
        result += next_counts * self.next_.count_possible_garden_plots_at_step(step)

        return result, full_area_count

    def _get_partial_square_area_results(self, step: int, full_area_count: int) -> int:
        result = 0
        if (
            full_area_count == 0
            and self.base.min_steps <= step
            and self.include_base_and_next
        ):
            result += self.base.count_possible_garden_plots_at_step(step)

        if (
            full_area_count <= 1
            and self.next_.min_steps <= step
            and self.include_base_and_next
        ):
            result += self.next_.count_possible_garden_plots_at_step(step)

        step_increase = self.next_.min_steps - self.base.min_steps

        full_followup_areas = max(0, full_area_count - 2)
        min_last_full_followup_area = (
            self.next_.min_steps + full_followup_areas * step_increase
        )
        partial_followup_areas = (step - min_last_full_followup_area) // step_increase

        if partial_followup_areas <= 0:
            return result

        for i in range(partial_followup_areas):
            step_increase_area = step_increase * (full_followup_areas + i + 1)

            counted_plots_for_partial_area = (
                self.next_.count_possible_garden_plots_at_step(
                    step - step_increase_area
                )
            )
            result += counted_plots_for_partial_area
        return result

    def count_possible_garden_plots(self, step: int) -> int:
        count, full_area_count = self._get_full_square_area_results(step)
        count += self._get_partial_square_area_results(step, full_area_count)
        return count


@dataclass(slots=True)
class _ExtendableArea:
    """Contains 4 quadrants.

    II  | I
    --- + --
    III | IV
    """

    quadrant1: _AreaSquareFull
    quadrant2: _AreaSquareFull
    quadrant3: _AreaSquareFull
    quadrant4: _AreaSquareFull

    base_quadrant: Literal[1, 2, 3, 4]

    def _get_quadrant_pair_for_row(
        self, row: Literal[0, 1]
    ) -> tuple[_AreaSquareFull, _AreaSquareFull]:
        return {
            1: [(self.quadrant1, self.quadrant2), (self.quadrant4, self.quadrant3)],
            2: [(self.quadrant2, self.quadrant1), (self.quadrant3, self.quadrant4)],
            3: [(self.quadrant3, self.quadrant4), (self.quadrant2, self.quadrant1)],
            4: [(self.quadrant4, self.quadrant3), (self.quadrant1, self.quadrant2)],
        }[self.base_quadrant][row]

    def _get_quadrant_pair_for_column(
        self, column: Literal[0, 1]
    ) -> tuple[_AreaSquareFull, _AreaSquareFull]:
        return {
            1: [(self.quadrant1, self.quadrant4), (self.quadrant2, self.quadrant3)],
            2: [(self.quadrant2, self.quadrant3), (self.quadrant1, self.quadrant4)],
            3: [(self.quadrant3, self.quadrant2), (self.quadrant4, self.quadrant1)],
            4: [(self.quadrant4, self.quadrant1), (self.quadrant3, self.quadrant2)],
        }[self.base_quadrant][column]

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
        pair_column_0 = self._get_quadrant_pair_for_column(0)
        for (base, next_), include_base_and_next in (
            (pair_row_0, True),
            (pair_row_1, True),
            (pair_column_0, False),
            (self._get_quadrant_pair_for_column(1), False),
        ):
            if base.min_steps > max_steps:
                continue
            yield _ExtendableDirection(
                base, next_, include_base_and_next=include_base_and_next
            )

        max_full_followup_areas = self._max_full_followup_areas_for_row(max_steps)
        if max_full_followup_areas <= 0:
            return

        step_increase = pair_row_0[1].min_steps - pair_row_0[0].min_steps

        def create_next_area_square(
            column_offset_from_base: int, row: Literal[0, 1]
        ) -> _AreaSquareExtended:
            offset_from_base = column_offset_from_base + row
            full_origin = pair_row_0[0] if offset_from_base % 2 == 0 else pair_row_1[0]

            row_origin = pair_row_0[0] if row == 0 else pair_row_1[0]
            next_step_increase = step_increase * column_offset_from_base
            return _AreaSquareExtended(
                min_steps=row_origin.min_steps + next_step_increase,
                max_steps=row_origin.max_steps + next_step_increase,
                full_origin=full_origin,
            )

        for i in range(max_full_followup_areas):
            next_row_0 = create_next_area_square(i + 2, 0)
            next_row_1 = create_next_area_square(i + 2, 1)

            yield _ExtendableDirection(
                next_row_0, next_row_1, include_base_and_next=False
            )


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
    completed_non_extendable_areas: list[_AreaSquareFull] = field(default_factory=list)

    """
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
    """

    def get_all_extendable_directions(
        self, max_steps: int
    ) -> Iterator[_ExtendableDirection]:
        for area in (
            self.bottom_right,
            self.top_left,
            self.top_right,
            self.bottom_left,
        ):
            yield from area.get_all_extendable_directions(max_steps)
        yield self.center_to_top
        yield self.center_to_right
        yield self.center_to_bottom
        yield self.center_to_left

    def get_non_extendable_area_results(self, step: int) -> Iterator[_AreaResult]:
        for area in self.completed_non_extendable_areas:
            yield _AreaResult(area.count_possible_garden_plots_at_step(step))


@dataclass(slots=True)
class _ResolverSquare:
    top_left: Coord2d
    width: int
    height: int
    visits_at_step: dict[int, dict[int, int]] = field(default_factory=dict, init=False)
    missing_visits: dict[int, set[int]] = field(init=False)
    _min_step: int | None = field(default=None, init=False)
    _max_step: int | None = field(default=None, init=False)

    map_: InitVar[_InfiniteMap]

    def __post_init__(self, map_: _InfiniteMap) -> None:
        self.missing_visits = {
            y: {x for x, symbol in x_symbol_iter if symbol == "."}
            for y, x_symbol_iter in map_.iter_data(
                self.top_left,
                Coord2d(
                    self.top_left.x + self.width - 1,
                    self.top_left.y + self.height - 1,
                ),
            )
        }

    @property
    def min_step(self) -> int:
        assert self._min_step is not None
        return self._min_step

    @property
    def max_step(self) -> int:
        assert self._max_step is not None
        return self._max_step

    @overload
    def is_inside(self, y: int) -> bool: ...

    @overload
    def is_inside(self, y: int, x: int) -> bool: ...

    def is_inside(self, y: int, x: int | None = None) -> bool:
        if y < self.top_left.y or y >= self.top_left.y + self.height:
            return False

        if x is None:
            return True

        return x >= self.top_left.x and x < self.top_left.x + self.width

    def add_visits(self, y: int, x_and_steps: Iterable[tuple[int, int]]) -> None:
        if y < self.top_left.y or y >= self.top_left.y + self.height:
            return

        for x, step in x_and_steps:
            if x < self.top_left.x or x >= self.top_left.x + self.width:
                continue

            visits_xs = self.visits_at_step.get(y)
            if visits_xs is None:
                visits_xs = {}
                self.visits_at_step[y] = visits_xs

            if (missing_xs := self.missing_visits.get(y)) and x in missing_xs:
                missing_xs.remove(x)
                if not missing_xs:
                    del self.missing_visits[y]
                visits_xs[x] = step
            else:
                prev_step = visits_xs[x]
                if prev_step > step:
                    visits_xs[x] = step

            if self._min_step is None or step < self._min_step:
                self._min_step = step
            if self._max_step is None or step > self._max_step:
                self._max_step = step

    def to_square(self, puzzle_max_steps: int) -> _AreaSquareFull:
        return _AreaSquareFull(
            self.min_step,
            self.max_step,
            puzzle_max_steps,
            Counter(
                step for xs in self.visits_at_step.values() for step in xs.values()
            ),
        )


class _ResolverQuadrants:
    def __init__(
        self, map_: _InfiniteMap, quadrant_from_center: Literal[1, 2, 3, 4]
    ) -> None:
        self.base_quadrant: Literal[1, 2, 3, 4]
        if quadrant_from_center == 1:
            top_left_coord = Coord2d(map_.last_x + 1, map_.first_y - 2 * map_.height)
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

        self.q2 = _ResolverSquare(top_left_coord, map_.width, map_.height, map_)
        self.q1 = _ResolverSquare(
            Coord2d(self.q2.top_left.x + map_.width, self.q2.top_left.y),
            map_.width,
            map_.height,
            map_,
        )
        self.q3 = _ResolverSquare(
            Coord2d(self.q2.top_left.x, self.q2.top_left.y + map_.height),
            map_.width,
            map_.height,
            map_,
        )
        self.q4 = _ResolverSquare(
            Coord2d(self.q3.top_left.x + map_.width, self.q3.top_left.y),
            map_.width,
            map_.height,
            map_,
        )

    @overload
    def is_inside(self, y: int) -> bool: ...
    @overload
    def is_inside(self, y: int, x: int) -> bool: ...

    def is_inside(self, y: int, x: int | None = None) -> bool:
        if y < self.q2.top_left.y or y >= self.q4.top_left.y + self.q4.height:
            return False

        if x is None:
            return True

        return x >= self.q2.top_left.x and x < self.q1.top_left.x + self.q1.width

    def add_visits(self, y: int, x_and_steps: Collection[tuple[int, int]]) -> None:
        if not self.is_inside(y):
            return

        for square in self.all_squares():
            square.add_visits(y, x_and_steps)

    def to_result(self, puzzle_max_steps: int) -> _ExtendableArea:
        return _ExtendableArea(
            self.q1.to_square(puzzle_max_steps),
            self.q2.to_square(puzzle_max_steps),
            self.q3.to_square(puzzle_max_steps),
            self.q4.to_square(puzzle_max_steps),
            self.base_quadrant,
        )

    def all_squares(self) -> Iterator[_ResolverSquare]:
        yield self.q1
        yield self.q2
        yield self.q3
        yield self.q4


class _ResolverVector:
    __slots__ = ("base", "next", "_min_y", "_max_y", "_min_x", "_max_x")

    @overload
    def __init__(
        self,
        map_: _InfiniteMap,
        center: _ResolverSquare,
        direction_from_center: CardinalDirection,
        /,
    ) -> None:
        pass

    @overload
    def __init__(self, base: _ResolverSquare, next_: _ResolverSquare, /) -> None:
        pass

    def __init__(
        self,
        map_or_base: _InfiniteMap | _ResolverSquare,
        next_or_center: _ResolverSquare,
        direction_from_center: CardinalDirection | None = None,
    ) -> None:
        if isinstance(map_or_base, _ResolverSquare):
            self.base = map_or_base
            assert isinstance(next_or_center, _ResolverSquare)
            self.next = next_or_center
            self._update_bounds()
            return

        map_ = map_or_base
        center = next_or_center
        assert direction_from_center is not None
        if direction_from_center == CardinalDirection.N:
            self.base = _ResolverSquare(
                Coord2d(center.top_left.x, center.top_left.y - map_.height),
                map_.width,
                map_.height,
                map_,
            )
            self.next = _ResolverSquare(
                Coord2d(self.base.top_left.x, self.base.top_left.y - map_.height),
                map_.width,
                map_.height,
                map_,
            )
        elif direction_from_center == CardinalDirection.S:
            self.base = _ResolverSquare(
                Coord2d(center.top_left.x, center.top_left.y + center.height),
                map_.width,
                map_.height,
                map_,
            )
            self.next = _ResolverSquare(
                Coord2d(self.base.top_left.x, self.base.top_left.y + self.base.height),
                map_.width,
                map_.height,
                map_,
            )
        elif direction_from_center == CardinalDirection.E:
            self.base = _ResolverSquare(
                Coord2d(center.top_left.x + center.width, center.top_left.y),
                map_.width,
                map_.height,
                map_,
            )
            self.next = _ResolverSquare(
                Coord2d(self.base.top_left.x + self.base.width, self.base.top_left.y),
                map_.width,
                map_.height,
                map_,
            )
        elif direction_from_center == CardinalDirection.W:
            self.base = _ResolverSquare(
                Coord2d(center.top_left.x - map_.width, center.top_left.y),
                map_.width,
                map_.height,
                map_,
            )
            self.next = _ResolverSquare(
                Coord2d(self.base.top_left.x - map_.width, self.base.top_left.y),
                map_.width,
                map_.height,
                map_,
            )
        else:
            assert_never(direction_from_center)
        self._update_bounds()

    def _update_bounds(self) -> None:
        self._min_y = min(self.base.top_left.y, self.next.top_left.y)
        self._max_y = max(
            self.base.top_left.y + self.base.height - 1,
            self.next.top_left.y + self.next.height - 1,
        )
        self._min_x = min(self.base.top_left.x, self.next.top_left.x)
        self._max_x = max(
            self.base.top_left.x + self.base.width - 1,
            self.next.top_left.x + self.next.width - 1,
        )

    def get_next(self, map_: _InfiniteMap) -> _ResolverVector:
        x_increase = self.next.top_left.x - self.base.top_left.x
        y_increase = self.next.top_left.y - self.base.top_left.y
        return _ResolverVector(
            self.next,
            _ResolverSquare(
                Coord2d(
                    self.next.top_left.x + x_increase,
                    self.next.top_left.y + y_increase,
                ),
                map_.width,
                map_.height,
                map_,
            ),
        )

    @overload
    def is_inside(self, y: int) -> bool: ...
    @overload
    def is_inside(self, y: int, x: int) -> bool: ...
    def is_inside(self, y: int, x: int | None = None) -> bool:
        min_y = min(self.base.top_left.y, self.next.top_left.y)
        if y < min_y:
            return False

        max_y = max(
            self.base.top_left.y + self.base.height - 1,
            self.next.top_left.y + self.next.height - 1,
        )
        if y > max_y:
            return False

        if x is None:
            return True

        return any(square.is_inside(y, x) for square in self.all_squares())

    def add_visits(self, y: int, x_and_steps: Collection[tuple[int, int]]) -> None:
        for square in self.all_squares():
            square.add_visits(y, x_and_steps)

    def is_extendable(self) -> bool:
        max_diff = self.next.max_step - self.base.max_step
        min_diff = self.next.min_step - self.base.min_step
        if max_diff != min_diff:
            return False

        y_increase = self.next.top_left.y - self.base.top_left.y
        x_increase = self.next.top_left.x - self.base.top_left.x

        prev_step_diff: int | None = None

        for (base_y, base_x_and_steps), (next_y, next_x_and_steps) in zip(
            sorted(self.base.visits_at_step.items(), key=lambda x: x[0]),
            sorted(self.next.visits_at_step.items(), key=lambda x: x[0]),
            strict=True,
        ):
            assert base_y + y_increase == next_y
            for (base_x, base_step), (next_x, next_step) in zip(
                sorted(base_x_and_steps.items(), key=lambda x: x[0]),
                sorted(next_x_and_steps.items(), key=lambda x: x[0]),
                strict=True,
            ):
                assert base_x + x_increase == next_x

                step_diff = next_step - base_step
                if prev_step_diff is None:
                    prev_step_diff = step_diff
                elif prev_step_diff != step_diff:
                    return False

        assert prev_step_diff is not None
        return True

    def to_result(self, puzzle_max_steps: int) -> _ExtendableDirection:
        return _ExtendableDirection(
            base=self.base.to_square(puzzle_max_steps),
            next_=self.next.to_square(puzzle_max_steps),
        )

    def all_squares(self) -> Iterator[_ResolverSquare]:
        yield self.base
        yield self.next


class _ResolverAroundStart:
    __slots__ = (
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "center_to_top",
        "center_to_right",
        "center_to_bottom",
        "center_to_left",
        "completed_non_extendable_areas",
        "_min_y",
        "_max_y",
        "_min_x",
        "_max_x",
    )

    def __init__(self, map_: _InfiniteMap) -> None:
        self.top_left = _ResolverQuadrants(map_, 2)
        self.top_right = _ResolverQuadrants(map_, 1)
        self.bottom_left = _ResolverQuadrants(map_, 3)
        self.bottom_right = _ResolverQuadrants(map_, 4)
        center = _ResolverSquare(
            Coord2d(map_.first_x, map_.first_y), map_.width, map_.height, map_
        )
        self.center_to_top = _ResolverVector(map_, center, CardinalDirection.N)
        self.center_to_right = _ResolverVector(map_, center, CardinalDirection.E)
        self.center_to_bottom = _ResolverVector(map_, center, CardinalDirection.S)
        self.center_to_left = _ResolverVector(map_, center, CardinalDirection.W)
        self.completed_non_extendable_areas = [center]

        self._update_bounds()

    def _update_bounds(self) -> None:
        self._min_y = min(
            s.top_left.y for s in (self.top_left.q2, self.center_to_top.next)
        )
        self._max_y = max(
            s.top_left.y + s.height - 1
            for s in (self.bottom_right.q4, self.center_to_bottom.next)
        )
        self._min_x = min(
            s.top_left.x for s in (self.top_left.q2, self.center_to_left.next)
        )
        self._max_x = max(
            s.top_left.x + s.width - 1
            for s in (self.bottom_right.q4, self.center_to_right.next)
        )

    @overload
    def is_inside(self, y: int) -> bool: ...

    @overload
    def is_inside(self, y: int, x: int) -> bool: ...

    def is_inside(self, y: int, x: int | None = None) -> bool:
        if y < self._min_y or y > self._max_y:
            return False

        if x is None:
            return True

        return any(area.is_inside(y, x) for area in self.all_areas())

    def add_visits(self, y: int, x_and_steps: Collection[tuple[int, int]]) -> None:
        for area in self.all_areas():
            area.add_visits(y, x_and_steps)

    def to_result(self, puzzle_max_steps: int) -> _AreasAroundStart:
        return _AreasAroundStart(
            top_left=self.top_left.to_result(puzzle_max_steps),
            top_right=self.top_right.to_result(puzzle_max_steps),
            bottom_left=self.bottom_left.to_result(puzzle_max_steps),
            bottom_right=self.bottom_right.to_result(puzzle_max_steps),
            center_to_top=self.center_to_top.to_result(puzzle_max_steps),
            center_to_right=self.center_to_right.to_result(puzzle_max_steps),
            center_to_bottom=self.center_to_bottom.to_result(puzzle_max_steps),
            center_to_left=self.center_to_left.to_result(puzzle_max_steps),
            completed_non_extendable_areas=[
                area.to_square(puzzle_max_steps)
                for area in self.completed_non_extendable_areas
            ],
        )

    def all_areas(
        self,
    ) -> Iterator[_ResolverSquare | _ResolverVector | _ResolverQuadrants]:
        yield self.top_left
        yield self.top_right
        yield self.bottom_left
        yield self.bottom_right
        yield self.center_to_top
        yield self.center_to_right
        yield self.center_to_bottom
        yield self.center_to_left
        yield from self.completed_non_extendable_areas

    def all_squares(self) -> Iterator[_ResolverSquare]:
        for area in self.all_areas():
            if isinstance(area, _ResolverSquare):
                yield area
            else:
                yield from area.all_squares()

    def all_vectors(self) -> Iterator[_ResolverVector]:
        yield self.center_to_top
        yield self.center_to_right
        yield self.center_to_bottom
        yield self.center_to_left

    def needs_processing(self) -> bool:
        return any(square.missing_visits for square in self.all_squares())

    def check_if_complete(self, map_: _InfiniteMap) -> list[_ResolverSquare]:
        new_squares: list[_ResolverSquare] = []
        if not self.center_to_top.is_extendable():
            self.completed_non_extendable_areas.append(self.center_to_top.base)
            self.center_to_top = self.center_to_top.get_next(map_)
            new_squares.append(self.center_to_top.next)
        if not self.center_to_right.is_extendable():
            self.completed_non_extendable_areas.append(self.center_to_right.base)
            self.center_to_right = self.center_to_right.get_next(map_)
            new_squares.append(self.center_to_right.next)
        if not self.center_to_bottom.is_extendable():
            self.completed_non_extendable_areas.append(self.center_to_bottom.base)
            self.center_to_bottom = self.center_to_bottom.get_next(map_)
            new_squares.append(self.center_to_bottom.next)
        if not self.center_to_left.is_extendable():
            self.completed_non_extendable_areas.append(self.center_to_left.base)
            self.center_to_left = self.center_to_left.get_next(map_)
            new_squares.append(self.center_to_left.next)
        if new_squares:
            self._update_bounds()
        return new_squares


def _is_garden_plot(x: int, map_row: _MapRow) -> bool:
    symbol = map_row[x]
    if symbol == "#":
        return False

    if symbol != ".":
        assert_never(symbol)

    return True


@dataclass(slots=True, frozen=True)
class _NeighborVisit:
    y_offset: Literal[-1, 0, 1]
    x_offset: Literal[-1, 0, 1]


def _get_neighbor_visits(
    x: int,
    neighbor_step: int,
    map_rows: list[_MapRow],
    known_steps_rows: list[dict[int, int]],
) -> Iterator[_NeighborVisit]:
    assert len(map_rows) == 3
    for y_offset, x_offset, data_ind in (
        (0, -1, 1),
        (0, 1, 1),
        (-1, 0, 0),
        (1, 0, 2),
    ):
        if (
            prev_step := known_steps_rows[data_ind].get(x + x_offset)
        ) is not None and prev_step <= neighbor_step:
            continue

        map_row = map_rows[data_ind]
        if _is_garden_plot(x + x_offset, map_row):
            assert y_offset in (-1, 0, 1)
            assert x_offset in (-1, 0, 1)
            yield _NeighborVisit(y_offset, x_offset)


class _ResolverInsidePoints:
    __slots__ = ("points", "_min_y", "_max_y", "_min_x", "_max_x")

    def __init__(self, resolver: _ResolverAroundStart) -> None:
        self.points: dict[int, set[int]] = {}
        for square in resolver.all_squares():
            for y, points in square.missing_visits.items():
                self.points.setdefault(y, set()).update(points)

        self._update_bounds()

    def _update_bounds(self) -> None:
        self._min_y = min(self.points)
        self._max_y = max(self.points)
        self._min_x = min(min(points) for points in self.points.values())
        self._max_x = max(max(points) for points in self.points.values())

    def add_square(self, square: _ResolverSquare) -> None:
        for y, points in square.missing_visits.items():
            self.points.setdefault(y, set()).update(points)
        self._update_bounds()

    # def _is_possibly_inside_by_bounds_y(self, y: int) -> bool:
    #    return self._min_y <= y <= self._max_y

    def is_possibly_inside(self, y: int) -> bool:
        # return self._is_possibly_inside_by_bounds_y(y)
        return y in self.points

    def is_inside(self, y: int, x: int) -> bool:
        #        if not self._is_possibly_inside_by_bounds_y(y):
        #            return False

        x_points = self.points.get(y)
        if x_points is None:
            return False
        return x in x_points


def _p2_step1_resolve_around_start(
    map_: _InfiniteMap,
    puzzle_max_steps: int,
) -> _AreasAroundStart:
    resolver = _ResolverAroundStart(map_)
    inside_points = _ResolverInsidePoints(resolver)

    ready_to_process: dict[int, list[tuple[int, int]]] = {
        map_.start.y: [(map_.start.x, 0)]
    }
    ready_to_process_outside: dict[int, list[tuple[int, int]]] = {}
    known_steps: dict[int, dict[int, int]] = {map_.start.y: {map_.start.x: 0}}

    resolver.add_visits(map_.start.y, [(map_.start.x, 0)])

    def get_or_create_known_step_row(
        known_steps: dict[int, dict[int, int]], y: int
    ) -> dict[int, int]:
        row = known_steps.get(y)
        if row is None:
            known_steps[y] = row = {}
        return row

    def get_or_create_row_list[T](container: dict[int, list[T]], y: int) -> list[T]:
        row = container.get(y)
        if row is None:
            container[y] = row = []
        return row

    while True:
        while ready_to_process:
            # if not resolver.needs_processing():
            #    break
            rows_to_process = list(ready_to_process.keys())
            for y in rows_to_process:
                x_and_step_to_process = ready_to_process.pop(y)
                if not x_and_step_to_process:
                    continue

                map_rows = [_MapRow(map_.get_row(y_)) for y_ in range(y - 1, y + 2)]
                known_steps_rows = [
                    get_or_create_known_step_row(known_steps, y_)
                    for y_ in range(y - 1, y + 2)
                ]

                new_locations_to_process: list[list[tuple[int, int]]] = [
                    [] for _ in range(3)
                ]
                for x, step in x_and_step_to_process:
                    neighbor_step = step + 1
                    for neighbor_visit in _get_neighbor_visits(
                        x, neighbor_step, map_rows, known_steps_rows
                    ):
                        neighbor_y_ind = neighbor_visit.y_offset + 1
                        neighbor_x = x + neighbor_visit.x_offset

                        known_steps_rows[neighbor_y_ind][neighbor_x] = neighbor_step
                        new_locations_to_process[neighbor_y_ind].append(
                            (neighbor_x, neighbor_step)
                        )

                for new_y_ind in range(3):
                    new_y = y + new_y_ind - 1
                    new_x_and_step_to_process = new_locations_to_process[new_y_ind]
                    if not new_x_and_step_to_process:
                        continue

                    if not inside_points.is_possibly_inside(new_y):
                        get_or_create_row_list(ready_to_process_outside, new_y).extend(
                            new_x_and_step_to_process
                        )
                        continue

                    ready_to_process_row = get_or_create_row_list(
                        ready_to_process, new_y
                    )
                    outside_to_process_row = get_or_create_row_list(
                        ready_to_process_outside, new_y
                    )
                    for new_x, step in new_x_and_step_to_process:
                        if not inside_points.is_inside(new_y, new_x):
                            outside_to_process_row.append((new_x, step))
                        else:
                            ready_to_process_row.append((new_x, step))
                    # resolver.add_visits(new_y, new_x_and_step_to_process)

        for y, x_and_step in known_steps.items():
            x_and_steps_list = list(x_and_step.items())
            resolver.add_visits(y, x_and_steps_list)
        new_squares = resolver.check_if_complete(map_)
        if not new_squares:
            break

        _logger.debug("New squares: %s", len(new_squares))
        # for y, x_and_step in known_steps.items():
        #    x_and_steps_list = list(x_and_step.items())
        #    for new_square in new_squares:
        #        new_square.add_visits(y, x_and_steps_list)
        for new_square in new_squares:
            inside_points.add_square(new_square)
            new_to_process = [
                (y, x_and_step)
                for y, x_and_step in ready_to_process_outside.items()
                if new_square.is_inside(y)
            ]
            for y, x_and_step in new_to_process:
                new_x_and_steps = [
                    (x, step) for x, step in x_and_step if new_square.is_inside(y, x)
                ]
                get_or_create_row_list(ready_to_process, y).extend(new_x_and_steps)
                # new_square.add_visits(y, new_x_and_steps)

                x_inside = {x for x, _ in new_x_and_steps}
                new_ready_to_process_outside_row = [
                    (x, step) for x, step in x_and_step if x not in x_inside
                ]
                ready_to_process_outside[y] = new_ready_to_process_outside_row

    return resolver.to_result(puzzle_max_steps)


def _p2_step2_extend_areas(areas: _AreasAroundStart, puzzle_max_steps: int) -> int:
    garden_plot_count = sum(
        area.possible_garden_plots
        for area in areas.get_non_extendable_area_results(puzzle_max_steps)
    )

    for extendable_direction in areas.get_all_extendable_directions(puzzle_max_steps):
        garden_plot_count += extendable_direction.count_possible_garden_plots(
            puzzle_max_steps
        )

    return garden_plot_count


REAL_STEPS = 26_501_365


def p2(input_str: str, puzzle_max_steps: int = 10_000_000) -> int:
    map_ = _InfiniteMap(input_str.splitlines())
    _logger.debug("Map size: w=%s, h=%s", map_.width, map_.height)
    assert map_.width % 2 == 1
    assert map_.height % 2 == 1

    # Step 1: START
    areas_around_start = _p2_step1_resolve_around_start(map_, puzzle_max_steps)

    # Step 1: END

    # Step 2: START
    return _p2_step2_extend_areas(areas_around_start, puzzle_max_steps)
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
