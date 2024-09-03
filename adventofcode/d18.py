import itertools
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from adventofcode.tooling.coordinates import Coord2d, X, Y
from adventofcode.tooling.directions import CardinalDirection as Dir

_logger = logging.getLogger(__name__)


@dataclass(slots=True, eq=False)
class _Path:
    corners: list[Coord2d] = field(
        default_factory=lambda: list[Coord2d]((Coord2d(Y(0), X(0)),))
    )
    _last_position: Coord2d = field(default_factory=lambda: Coord2d(Y(0), X(0)))
    _north_west_corner: Coord2d = field(default_factory=lambda: Coord2d(Y(0), X(0)))
    _south_east_corner: Coord2d = field(default_factory=lambda: Coord2d(Y(0), X(0)))

    @property
    def height(self) -> int:
        assert self._north_west_corner == Coord2d(Y(0), X(0))
        return self._south_east_corner.y + 1

    @property
    def width(self) -> int:
        assert self._north_west_corner == Coord2d(Y(0), X(0))
        return self._south_east_corner.x + 1

    def process(self, dir_counts: Iterable[tuple[Dir, int]]) -> None:
        for direction, count in dir_counts:
            self.add(direction, count)

    def add(self, direction: Dir, count: int) -> None:
        self._last_position = self._last_position.get_relative(direction, count)
        self.corners.append(self._last_position)

        self._north_west_corner = Coord2d(
            min(self._north_west_corner.y, self._last_position.y),
            min(self._north_west_corner.x, self._last_position.x),
        )
        self._south_east_corner = Coord2d(
            max(self._south_east_corner.y, self._last_position.y),
            max(self._south_east_corner.x, self._last_position.x),
        )
        _logger.debug("Adding %s, %s -> %s", direction, count, self._last_position)

    def normalize(self) -> None:
        assert self.corners
        assert self.corners[0] == self.corners[-1]
        del self.corners[-1]
        x_adjustment = -self._north_west_corner.x
        y_adjustment = -self._north_west_corner.y
        self.corners = [
            Coord2d(Y(coord.y + y_adjustment), X(coord.x + x_adjustment))
            for coord in self.corners
        ]
        self._north_west_corner = Coord2d(Y(0), X(0))
        self._south_east_corner = Coord2d(
            Y(self._south_east_corner.y + y_adjustment),
            X(self._south_east_corner.x + x_adjustment),
        )
        if _logger.isEnabledFor(logging.DEBUG):
            path = " -> ".join(str(coord) for coord in self.corners)
            _logger.debug("Normalized path: %s", path)


@dataclass
class _Line:
    c1: Coord2d
    c2: Coord2d


def _find_at_row_from_sorted(line_list: Iterable[_Line], row: int) -> Iterable[_Line]:
    for line in line_list:
        if line.c1.y > row:
            break
        if row == line.c1.y:
            yield line


@dataclass(slots=True)
class _PathLines:
    verticals: list[_Line]
    horizontals: list[_Line]
    flats_within_verticals: list[_Line]
    height: int
    width: int

    def __init__(self, path: _Path) -> None:
        self.flats_within_verticals = []
        self.horizontals = []
        self.verticals = []
        self.height = path.height
        self.width = path.width

        prevs: list[Coord2d] = path.corners[-3:]
        for corner in path.corners:
            direction = prevs[-1].dir_to(corner)
            if direction in (Dir.N, Dir.S):
                self.verticals.append(_Line(prevs[-1], corner))
                assert len(prevs) >= 3
                prev_dir = prevs[-2].dir_to(prevs[-1])
                assert prev_dir in (Dir.E, Dir.W)
                prev2_dir = prevs[-3].dir_to(prevs[-2])
                if prev2_dir == direction:
                    self.flats_within_verticals.append(_Line(prevs[-2], prevs[-1]))
                else:
                    assert direction in (Dir.N, Dir.S)
                    assert prev2_dir == direction.opposite()
                    self.horizontals.append(_Line(prevs[-2], prevs[-1]))
            else:
                assert direction in (Dir.E, Dir.W)
            prevs = prevs[-2:]
            prevs.append(corner)

    def optimize(self) -> None:
        def _smaller_x_first(flat_list: Iterable[_Line]) -> Iterable[_Line]:
            for line in flat_list:
                if line.c1.x < line.c2.x:
                    yield line
                else:
                    yield _Line(line.c2, line.c1)

        self.flats_within_verticals = list(
            _smaller_x_first(self.flats_within_verticals)
        )
        self.flats_within_verticals.sort(key=lambda line: line.c1.y)

        self.horizontals = list(_smaller_x_first(self.horizontals))
        self.horizontals.sort(key=lambda line: line.c1.y)

        self.verticals = [
            line if line.c1.y < line.c2.y else _Line(line.c2, line.c1)
            for line in self.verticals
        ]
        self.verticals.sort(key=lambda line: line.c1.y)

    def count_inside_columns(self, row: int) -> int:
        assert 0 <= row < self.height

        flats = list(_find_at_row_from_sorted(self.flats_within_verticals, row))
        flats_firsts = {line.c1.x for line in flats}
        flats_lasts = {line.c2.x for line in flats}
        horizontals = list(_find_at_row_from_sorted(self.horizontals, row))
        horizontals_firsts = {line.c1.x for line in horizontals}
        horizontals_lasts = {line.c2.x for line in horizontals}

        def get_crossing_vertice_columns_set() -> set[X]:
            crossing_vertice_columns_set = set[X]()
            for line in self.verticals:
                assert line.c1.x == line.c2.x
                if line.c1.y >= row:
                    break
                if row <= line.c1.y or line.c2.y <= row:
                    continue
                crossing_vertice_columns_set.add(line.c1.x)
            return crossing_vertice_columns_set

        crossing_vertice_columns_set = get_crossing_vertice_columns_set()

        def assert_disjoints() -> None:
            if not __debug__:
                return
            disjoint_sets = [
                crossing_vertice_columns_set,
                horizontals_firsts,
                horizontals_lasts,
                flats_firsts,
                flats_lasts,
            ]
            for set1, set2 in itertools.combinations(disjoint_sets, 2):
                assert set1.isdisjoint(set2)

        assert_disjoints()

        columns_to_check = list[int](crossing_vertice_columns_set)
        columns_to_check.extend(x for line in flats for x in (line.c1.x, line.c2.x))
        columns_to_check.extend(
            x for line in horizontals for x in (line.c1.x, line.c2.x)
        )
        columns_to_check.sort()

        @dataclass
        class ProcessData:
            count: int = 0
            x_inside_first: int | None = None
            line_in_progress: _Line | None = None

            def process_column(self, x: int) -> None:
                if self.x_inside_first is None:
                    assert x not in flats_lasts
                    assert x not in horizontals_lasts
                    self.x_inside_first = x
                    if x in crossing_vertice_columns_set:
                        pass
                    elif x in flats_firsts:
                        self.line_in_progress = next(
                            line for line in flats if line.c1.x == x
                        )
                    elif x in horizontals_firsts:
                        self.line_in_progress = next(
                            line for line in horizontals if line.c1.x == x
                        )

                elif x in crossing_vertice_columns_set:
                    assert self.line_in_progress is None
                    self.count += x - self.x_inside_first + 1
                    self.x_inside_first = None

                elif x in horizontals_firsts:
                    assert self.line_in_progress is None
                    self.line_in_progress = next(
                        line for line in horizontals if line.c1.x == x
                    )

                elif x in horizontals_lasts:
                    assert self.line_in_progress is not None
                    assert self.line_in_progress.c2.x == x

                    if self.line_in_progress.c1.x == self.x_inside_first:
                        self.count += x - self.x_inside_first + 1
                        self.x_inside_first = None
                    self.line_in_progress = None

                elif x in flats_firsts:
                    assert self.line_in_progress is None
                    self.line_in_progress = next(
                        line for line in flats if line.c1.x == x
                    )

                elif x in flats_lasts:
                    assert self.line_in_progress is not None
                    assert self.line_in_progress.c2.x == x

                    if self.line_in_progress.c1.x != self.x_inside_first:
                        self.count += x - self.x_inside_first + 1
                        self.x_inside_first = None
                    self.line_in_progress = None

        data = ProcessData()
        for x in columns_to_check:
            data.process_column(x)

        return data.count


@dataclass
class _InsideCountGroups:
    unique_rows: dict[int, int]
    row_ranges: list[tuple[int, int, int]]

    def __init__(self, segments: _PathLines) -> None:
        self.unique_rows = {}
        self.row_ranges = []

        rows_with_flats = set[int]()
        for line in itertools.chain(
            segments.flats_within_verticals, segments.horizontals
        ):
            rows_with_flats.add(line.c1.y)
            rows_with_flats.add(line.c2.y)
        rows_with_unique_counts = set[int](
            y for line in segments.verticals for y in (line.c1.y, line.c2.y)
        )
        assert rows_with_unique_counts == rows_with_flats

        rows_to_check = list[int](rows_with_unique_counts)
        rows_to_check.sort()
        assert rows_to_check[0] == 0
        assert rows_to_check[-1] == segments.height - 1

        _logger.debug("Rows to check: %s", rows_to_check)

        prev_line_with_vertices_only: tuple[int, int] | None = None
        for y in rows_to_check:
            self.unique_rows[y] = segments.count_inside_columns(y)
            if prev_line_with_vertices_only is not None:
                assert prev_line_with_vertices_only[0] < y
                assert y - 1 not in rows_with_unique_counts
                self.row_ranges.append(
                    (
                        prev_line_with_vertices_only[0],
                        y - 1,
                        prev_line_with_vertices_only[1],
                    )
                )
                prev_line_with_vertices_only = None

            if y + 1 == segments.height or y + 1 in rows_with_unique_counts:
                continue

            assert prev_line_with_vertices_only is None
            prev_line_with_vertices_only = (y + 1, segments.count_inside_columns(y + 1))

        assert prev_line_with_vertices_only is None

    def total_inside_positions(self) -> int:
        return sum(self.unique_rows.values()) + sum(
            count * (y_last - y_first + 1) for y_first, y_last, count in self.row_ranges
        )


def _resolve(dir_counts: Iterable[tuple[Dir, int]]) -> int:
    path = _Path()
    path.process(dir_counts)
    path.normalize()

    segments = _PathLines(path)
    segments.optimize()
    segment_groups = _InsideCountGroups(segments)
    return segment_groups.total_inside_positions()


def p1(input_str: str) -> int:
    def _direction_to_dir(direction: str) -> Dir:
        return {"U": Dir.N, "R": Dir.E, "D": Dir.S, "L": Dir.W}[direction]

    def _parse_input(lines: Iterable[str]) -> Iterable[tuple[Dir, int]]:
        for line in lines:
            direction, count, _ = line.split()
            yield _direction_to_dir(direction), int(count)

    return _resolve(_parse_input(input_str.splitlines()))


def p2(input_str: str) -> int:
    def _direction_to_dir(direction: str) -> Dir:
        return {"0": Dir.E, "1": Dir.S, "2": Dir.W, "3": Dir.N}[direction]

    def _parse_input(lines: Iterable[str]) -> Iterable[tuple[Dir, int]]:
        for line in lines:
            _, _, hex_part = line.split()
            count = int(hex_part[2:-2], base=16)
            direction = _direction_to_dir(hex_part[-2])
            yield direction, count

    return _resolve(_parse_input(input_str.splitlines()))
