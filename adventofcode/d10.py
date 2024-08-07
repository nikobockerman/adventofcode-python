import enum
import itertools
import logging
from dataclasses import dataclass, field
from typing import Iterable

from adventofcode.tooling.directions import CardinalDirection as Dir
from adventofcode.tooling.directions import CardinalDirectionsAll
from adventofcode.tooling.map import Coord2d, Map2d

_logger = logging.getLogger(__name__)


class _Inside(enum.Enum):
    Inside = enum.auto()
    Outside = enum.auto()
    InPath = enum.auto()
    Unknown = enum.auto()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


@dataclass
class _Pipe:
    coord: Coord2d
    symbol: str
    inside: _Inside = _Inside.Unknown

    def __hash__(self) -> int:
        return hash(self.coord)


_symbols_open_to_north = "|JLS"
_symbols_open_to_east = "-FLS"
_symbols_open_to_south = "|F7S"
_symbols_open_to_west = "-J7S"


@dataclass
class _MapData(Map2d[_Pipe]):
    start: _Pipe

    def __init__(self, input_data: list[str]) -> None:
        data: list[list[_Pipe]] = []
        start: _Pipe | None = None
        for y, row in enumerate(input_data):
            y_pipes: list[_Pipe] = []
            data.append(y_pipes)
            for x, symbol in enumerate(row):
                pipe = _Pipe(Coord2d(x, y), symbol)
                if symbol == "S":
                    assert start is None
                    start = pipe
                y_pipes.append(pipe)
        super().__init__(data)
        assert start is not None
        self.start = start


def _get_adjoin_pipes_on_path(pipe: _Pipe, map_data: _MapData) -> Iterable[_Pipe]:
    east = map_data.get(pipe.coord.adjoin(Dir.E), None)
    if (
        east
        and pipe.symbol in _symbols_open_to_east
        and east.symbol in _symbols_open_to_west
    ):
        yield east

    south = map_data.get(pipe.coord.adjoin(Dir.S), None)
    if (
        south
        and pipe.symbol in _symbols_open_to_south
        and south.symbol in _symbols_open_to_north
    ):
        yield south

    west = map_data.get(pipe.coord.adjoin(Dir.W), None)
    if (
        west
        and pipe.symbol in _symbols_open_to_west
        and west.symbol in _symbols_open_to_east
    ):
        yield west

    north = map_data.get(pipe.coord.adjoin(Dir.N), None)
    if (
        north
        and pipe.symbol in _symbols_open_to_north
        and north.symbol in _symbols_open_to_south
    ):
        yield north


def p1(input_str: str) -> int:
    map_data = _MapData(list(input_str.splitlines()))

    start = map_data.start
    neighbors_cur = list(_get_adjoin_pipes_on_path(start, map_data))
    assert len(neighbors_cur) == 2

    neighbors_prev = [start, start]

    dist = 1
    while neighbors_cur[0] != neighbors_cur[1]:
        neighbors_new = [
            next(n for n in _get_adjoin_pipes_on_path(cur, map_data) if n != prev)
            for prev, cur in zip(neighbors_prev, neighbors_cur)
        ]
        neighbors_prev = neighbors_cur
        neighbors_cur = neighbors_new
        dist += 1

    return dist


def p2(input_str: str) -> int:
    map_data = _MapData(list(input_str.splitlines()))

    start = map_data.start
    start_neighbors = list(_get_adjoin_pipes_on_path(start, map_data))

    def resolve_start_symbol() -> str:
        north = start.coord.adjoin(Dir.N)
        west = start.coord.adjoin(Dir.W)
        south = start.coord.adjoin(Dir.S)
        east = start.coord.adjoin(Dir.E)
        start_neighbor_coords = frozenset(n.coord for n in start_neighbors)
        return {
            frozenset((north, south)): "|",
            frozenset((east, west)): "-",
            frozenset((north, east)): "J",
            frozenset((north, west)): "L",
            frozenset((south, east)): "7",
            frozenset((south, west)): "F",
        }[start_neighbor_coords]

    fixed_start_symbol = resolve_start_symbol()
    _logger.debug("fixed_start_symbol=%s", fixed_start_symbol)
    start.symbol = fixed_start_symbol

    path_by_pipes: list[_Pipe] = [start, start_neighbors[0]]
    while True:
        n = next(
            pipe
            for pipe in _get_adjoin_pipes_on_path(path_by_pipes[-1], map_data)
            if pipe is not path_by_pipes[-2]
        )
        if n == start:
            break
        path_by_pipes.append(n)

    for p in path_by_pipes:
        p.inside = _Inside.InPath

    coords_in_path = set(pipe.coord for pipe in path_by_pipes)

    @dataclass
    class PathPipe:
        pipe: _Pipe
        neighbors: dict[Dir, _Inside]

    def create_first_path_pipe() -> PathPipe:
        # Guessed value for y to hit path on | symbol
        y = (map_data.height // 2) + 1

        for _, x_iter in map_data.iter_data(
            Coord2d(0, y), Coord2d(map_data.last_x, y + 1)
        ):
            for _, pipe in x_iter:
                coord = pipe.coord
                if coord not in coords_in_path:
                    assert pipe.inside is _Inside.Unknown
                    pipe.inside = _Inside.Outside
                else:
                    assert pipe.symbol not in "J7-S"
                    assert pipe.symbol == "|", "Use better value for y"
                    east_neighbor = map_data.get(pipe.coord.adjoin(Dir.E))
                    if east_neighbor and east_neighbor.coord not in coords_in_path:
                        east_neighbor.inside = _Inside.Inside
                    return PathPipe(
                        pipe, {Dir.E: _Inside.Inside, Dir.W: _Inside.Outside}
                    )
        raise AssertionError()

    first_path_pipe = create_first_path_pipe()
    _logger.debug(f"{first_path_pipe=}")

    @dataclass
    class NeighborCheckGroup:
        prev_neighbor_dirs_to_check: list[Dir]
        neighbor_dirs_to_set: list[Dir] = field(default_factory=list)
        inside: _Inside = _Inside.Unknown

    def create_path_pipe(prev: PathPipe, pipe: _Pipe) -> PathPipe:
        path_dir = prev.pipe.coord.dir_to(pipe.coord)

        neighbor_check_groups: list[NeighborCheckGroup] = []
        if pipe.symbol == "|":
            neighbor_check_groups = [
                NeighborCheckGroup([Dir.W], [Dir.W]),
                NeighborCheckGroup([Dir.E], [Dir.E]),
            ]
        elif pipe.symbol == "-":
            neighbor_check_groups = [
                NeighborCheckGroup([Dir.N], [Dir.N]),
                NeighborCheckGroup([Dir.S], [Dir.S]),
            ]
        elif pipe.symbol == "L":
            if path_dir is Dir.W:
                neighbor_check_groups = [
                    NeighborCheckGroup([Dir.S], [Dir.S, Dir.W]),
                    NeighborCheckGroup([Dir.N]),
                ]
            elif path_dir is Dir.S:
                neighbor_check_groups = [
                    NeighborCheckGroup([Dir.W], [Dir.S, Dir.W]),
                    NeighborCheckGroup([Dir.E]),
                ]
            else:
                raise AssertionError()
        elif pipe.symbol == "F":
            if path_dir is Dir.N:
                neighbor_check_groups = [
                    NeighborCheckGroup([Dir.W], [Dir.N, Dir.W]),
                    NeighborCheckGroup([Dir.E]),
                ]
            elif path_dir is Dir.W:
                neighbor_check_groups = [
                    NeighborCheckGroup([Dir.N], [Dir.N, Dir.W]),
                    NeighborCheckGroup([Dir.S]),
                ]
            else:
                raise AssertionError()
        elif pipe.symbol == "7":
            if path_dir is Dir.N:
                neighbor_check_groups = [
                    NeighborCheckGroup([Dir.E], [Dir.N, Dir.E]),
                    NeighborCheckGroup([Dir.W]),
                ]
            elif path_dir is Dir.E:
                neighbor_check_groups = [
                    NeighborCheckGroup([Dir.N], [Dir.N, Dir.E]),
                    NeighborCheckGroup([Dir.S]),
                ]
            else:
                raise AssertionError()
        elif pipe.symbol == "J":
            if path_dir is Dir.S:
                neighbor_check_groups = [
                    NeighborCheckGroup([Dir.E], [Dir.S, Dir.E]),
                    NeighborCheckGroup([Dir.W]),
                ]
            elif path_dir is Dir.E:
                neighbor_check_groups = [
                    NeighborCheckGroup([Dir.S], [Dir.S, Dir.E]),
                    NeighborCheckGroup([Dir.N]),
                ]
            else:
                raise AssertionError()
        else:
            raise AssertionError()

        assert len(neighbor_check_groups) == 2

        # Determine inside/outside for check groups
        for check_group in neighbor_check_groups:
            for check_dir in check_group.prev_neighbor_dirs_to_check:
                prev_inside = prev.neighbors.get(check_dir)
                if prev_inside is None:
                    continue
                assert prev_inside in (_Inside.Inside, _Inside.Outside)
                check_group.inside = prev_inside
                break

        # If any check groups are still unknown, set them to the opposite of the known
        unknown_check_groups_with_neighbors_to_set = [
            check_group
            for check_group in neighbor_check_groups
            if check_group.inside is _Inside.Unknown
            and check_group.neighbor_dirs_to_set
        ]
        assert len(unknown_check_groups_with_neighbors_to_set) <= 1
        if unknown_check_groups_with_neighbors_to_set:
            known_check_groups = [
                check_group
                for check_group in neighbor_check_groups
                if check_group.inside is not _Inside.Unknown
            ]
            assert len(known_check_groups) == 1
            known_inside = known_check_groups[0].inside
            inside_for_unknown = (
                _Inside.Inside if known_inside is _Inside.Outside else _Inside.Outside
            )
            for check_group in unknown_check_groups_with_neighbors_to_set:
                check_group.inside = inside_for_unknown

        # Record neighbors for PathPipe to be used on next iteration
        neighbors = {
            neighbor_dir: check_group.inside
            for check_group in neighbor_check_groups
            for neighbor_dir in check_group.neighbor_dirs_to_set
        }
        assert all(
            value in (_Inside.Inside, _Inside.Outside) for value in neighbors.values()
        )

        # Set inside/outside for direct neighbors already in map as we have the data
        # available
        for neighbor_dir, inside in neighbors.items():
            neighbor = map_data.get(pipe.coord.adjoin(neighbor_dir), None)
            if neighbor is not None and neighbor.inside is not _Inside.InPath:
                assert neighbor.inside is _Inside.Unknown or neighbor.inside is inside
                neighbor.inside = inside

        return PathPipe(pipe, neighbors)

    _logger.info("Detecting inside/outside neighbors along path")
    index_in_path_for_first_path_pipe = path_by_pipes.index(first_path_pipe.pipe)
    prev_path_pipe: PathPipe = first_path_pipe
    for pipe in itertools.chain(
        path_by_pipes[index_in_path_for_first_path_pipe + 1 :],
        path_by_pipes[:index_in_path_for_first_path_pipe],
    ):
        prev_path_pipe = create_path_pipe(prev_path_pipe, pipe)

    _logger.info("Marking rest of map for inside/outside")

    def mark_pipe(
        pipe: _Pipe, visited_recursive_coords: set[Coord2d] | None = None
    ) -> None:
        if pipe.inside is not _Inside.Unknown:
            return

        if visited_recursive_coords is None:
            visited_recursive_coords = set()
        visited_recursive_coords.add(pipe.coord)

        for neighbor in (
            map_data.get(pipe.coord.adjoin(direction), None)
            for direction in CardinalDirectionsAll
        ):
            if not neighbor:
                continue
            if neighbor.coord in visited_recursive_coords:
                continue
            mark_pipe(neighbor, visited_recursive_coords)
            if neighbor.inside in (_Inside.Inside, _Inside.Outside):
                pipe.inside = neighbor.inside
                return
        raise AssertionError()

    for _, pipe_iter in map_data.iter_data():
        for _, pipe in pipe_iter:
            mark_pipe(pipe)

    def get_symbol_for_pipe(pipe: _Pipe) -> str:
        if pipe.inside is _Inside.Inside:
            return " "
        if pipe.inside is _Inside.Outside:
            return "."
        if pipe.inside is _Inside.InPath:
            return pipe.symbol
        return "#"

    if _logger.isEnabledFor(logging.DEBUG):
        for map_line in map_data.str_lines(get_symbol_for_pipe):
            print(map_line)

    _logger.info("Calculating inside locations")

    return sum(
        1
        for _, pipe_iter in map_data.iter_data()
        for _, pipe in pipe_iter
        if pipe.inside is _Inside.Inside
    )
