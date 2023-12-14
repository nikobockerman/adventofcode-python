import enum
import itertools
import logging
from dataclasses import dataclass, field
from typing import Iterable

from adventofcode.tooling.map import Coord2d, Dir, Map2d

logger = logging.getLogger(__name__)


class Inside(enum.Enum):
    Inside = enum.auto()
    Outside = enum.auto()
    InPath = enum.auto()
    Unknown = enum.auto()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


@dataclass
class Pipe:
    coord: Coord2d
    symbol: str
    inside: Inside = Inside.Unknown

    def __hash__(self) -> int:
        return hash(self.coord)


symbols_open_to_north = "|JLS"
symbols_open_to_east = "-FLS"
symbols_open_to_south = "|F7S"
symbols_open_to_west = "-J7S"


@dataclass
class MapPipe:
    pipe: Pipe
    neighbors: dict[Dir, "MapPipe"] = field(default_factory=dict)

    def __get_neighbors_str(self) -> Iterable[str]:
        for direction, neighbor in self.neighbors.items():
            yield f"{direction.name}: {neighbor.pipe.coord}"

    def __repr__(self) -> str:
        neighbors_str = "{" + ",".join(self.__get_neighbors_str()) + "}"
        return f"MapPipe(pipe={self.pipe}, neighbors={neighbors_str})"

    def get_adjoin_pipes_on_path(self) -> Iterable["MapPipe"]:
        east = self.neighbors.get(Dir.E)
        if (
            east
            and self.pipe.symbol in symbols_open_to_east
            and east.pipe.symbol in symbols_open_to_west
        ):
            yield east

        south = self.neighbors.get(Dir.S)
        if (
            south
            and self.pipe.symbol in symbols_open_to_south
            and south.pipe.symbol in symbols_open_to_north
        ):
            yield south

        west = self.neighbors.get(Dir.W)
        if (
            west
            and self.pipe.symbol in symbols_open_to_west
            and west.pipe.symbol in symbols_open_to_east
        ):
            yield west

        north = self.neighbors.get(Dir.N)
        if (
            north
            and self.pipe.symbol in symbols_open_to_north
            and north.pipe.symbol in symbols_open_to_south
        ):
            yield north


@dataclass
class MapData(Map2d[MapPipe]):
    start: MapPipe

    def __init__(self, input_data: list[str]) -> None:
        data: list[list[MapPipe]] = []
        start: MapPipe | None = None
        for y, row in enumerate(input_data):
            y_pipes: list[MapPipe] = []
            data.append(y_pipes)
            for x, symbol in enumerate(row):
                pipe = Pipe(Coord2d(x, y), symbol)
                map_pipe = MapPipe(pipe)
                if symbol == "S":
                    assert start is None
                    start = map_pipe
                if x > 0:
                    east_map_pipe = y_pipes[-1]
                    map_pipe.neighbors[Dir.W] = east_map_pipe
                    east_map_pipe.neighbors[Dir.E] = map_pipe
                if y > 0:
                    north_map_pipe = data[y - 1][x]
                    map_pipe.neighbors[Dir.N] = north_map_pipe
                    north_map_pipe.neighbors[Dir.S] = map_pipe
                y_pipes.append(map_pipe)
        super().__init__(data)
        assert start is not None
        self.start = start


def p1(input_str: str) -> int:
    map_data = MapData(list(input_str.splitlines()))

    start = map_data.start
    neighbors_cur = list(start.get_adjoin_pipes_on_path())
    assert len(neighbors_cur) == 2

    neighbors_prev = [start, start]

    dist = 1
    while neighbors_cur[0] != neighbors_cur[1]:
        neighbors_new = [
            next(n for n in cur.get_adjoin_pipes_on_path() if n != prev)
            for prev, cur in zip(neighbors_prev, neighbors_cur)
        ]
        neighbors_prev = neighbors_cur
        neighbors_cur = neighbors_new
        dist += 1

    return dist


def p2(input_str: str) -> int:
    map_data = MapData(list(input_str.splitlines()))

    start = map_data.start
    start_neighbors = list(start.get_adjoin_pipes_on_path())

    def resolve_start_symbol() -> str:
        def get_neigbor_coord(direction: Dir) -> Coord2d:
            pipe = start.neighbors.get(direction)
            assert pipe is not None
            return pipe.pipe.coord

        north = get_neigbor_coord(Dir.N)
        west = get_neigbor_coord(Dir.W)
        south = get_neigbor_coord(Dir.S)
        east = get_neigbor_coord(Dir.E)
        start_neighbor_coords = frozenset(n.pipe.coord for n in start_neighbors)
        return {
            frozenset((north, south)): "|",
            frozenset((east, west)): "-",
            frozenset((north, east)): "J",
            frozenset((north, west)): "L",
            frozenset((south, east)): "7",
            frozenset((south, west)): "F",
        }[start_neighbor_coords]

    fixed_start_symbol = resolve_start_symbol()
    logger.debug("fixed_start_symbol=%s", fixed_start_symbol)
    start.pipe.symbol = fixed_start_symbol

    path_by_pipes: list[MapPipe] = [start, start_neighbors[0]]
    while True:
        n = next(
            pipe
            for pipe in path_by_pipes[-1].get_adjoin_pipes_on_path()
            if pipe is not path_by_pipes[-2]
        )
        if n == start:
            break
        path_by_pipes.append(n)

    for p in path_by_pipes:
        p.pipe.inside = Inside.InPath

    coords_in_path = set(pipe.pipe.coord for pipe in path_by_pipes)

    @dataclass
    class PathPipe:
        pipe: MapPipe
        neighbors: dict[Dir, Inside]

    def create_first_path_pipe() -> PathPipe:
        # Guessed value for y to hit path on | symbol
        y = (map_data.len_y // 2) + 1

        for _, map_pipe in map_data.iter_data(
            start=Coord2d(0, y), stop=Coord2d(map_data.len_x, y + 1)
        ):
            coord = map_pipe.pipe.coord
            if coord not in coords_in_path:
                assert map_pipe.pipe.inside is Inside.Unknown
                map_pipe.pipe.inside = Inside.Outside
            else:
                assert map_pipe.pipe.symbol not in "J7-S"
                assert map_pipe.pipe.symbol == "|", "Use better value for y"
                east_neighbor = map_pipe.neighbors.get(Dir.E)
                if east_neighbor and east_neighbor.pipe.coord not in coords_in_path:
                    east_neighbor.pipe.inside = Inside.Inside
                return PathPipe(map_pipe, {Dir.E: Inside.Inside, Dir.W: Inside.Outside})
        raise AssertionError()

    first_path_pipe = create_first_path_pipe()
    logger.debug(f"{first_path_pipe=}")

    @dataclass
    class NeighborCheckGroup:
        prev_neighbor_dirs_to_check: list[Dir]
        neighbor_dirs_to_set: list[Dir] = field(default_factory=list)
        inside: Inside = Inside.Unknown

    def create_path_pipe(prev: PathPipe, map_pipe: MapPipe) -> PathPipe:
        path_dir = prev.pipe.pipe.coord.dir_to(map_pipe.pipe.coord)

        neighbor_check_groups: list[NeighborCheckGroup] = []
        if map_pipe.pipe.symbol == "|":
            neighbor_check_groups = [
                NeighborCheckGroup([Dir.W], [Dir.W]),
                NeighborCheckGroup([Dir.E], [Dir.E]),
            ]
        elif map_pipe.pipe.symbol == "-":
            neighbor_check_groups = [
                NeighborCheckGroup([Dir.N], [Dir.N]),
                NeighborCheckGroup([Dir.S], [Dir.S]),
            ]
        elif map_pipe.pipe.symbol == "L":
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
        elif map_pipe.pipe.symbol == "F":
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
        elif map_pipe.pipe.symbol == "7":
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
        elif map_pipe.pipe.symbol == "J":
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
                assert prev_inside in (Inside.Inside, Inside.Outside)
                check_group.inside = prev_inside
                break

        # If any check groups are still unknown, set them to the opposite of the known
        unknown_check_groups_with_neighbors_to_set = [
            check_group
            for check_group in neighbor_check_groups
            if check_group.inside is Inside.Unknown and check_group.neighbor_dirs_to_set
        ]
        assert len(unknown_check_groups_with_neighbors_to_set) <= 1
        if unknown_check_groups_with_neighbors_to_set:
            known_check_groups = [
                check_group
                for check_group in neighbor_check_groups
                if check_group.inside is not Inside.Unknown
            ]
            assert len(known_check_groups) == 1
            known_inside = known_check_groups[0].inside
            inside_for_unknown = (
                Inside.Inside if known_inside is Inside.Outside else Inside.Outside
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
            value in (Inside.Inside, Inside.Outside) for value in neighbors.values()
        )

        # Set inside/outside for direct neighbors already in map as we have the data
        # available
        for neighbor_dir, inside in neighbors.items():
            neighbor = map_pipe.neighbors.get(neighbor_dir)
            if neighbor is not None and neighbor.pipe.inside is not Inside.InPath:
                assert (
                    neighbor.pipe.inside is Inside.Unknown
                    or neighbor.pipe.inside is inside
                )
                neighbor.pipe.inside = inside

        return PathPipe(map_pipe, neighbors)

    logger.info("Detecting inside/outside neighbors along path")
    index_in_path_for_first_path_pipe = path_by_pipes.index(first_path_pipe.pipe)
    prev_path_pipe: PathPipe = first_path_pipe
    for map_pipe in itertools.chain(
        path_by_pipes[index_in_path_for_first_path_pipe + 1 :],
        path_by_pipes[:index_in_path_for_first_path_pipe],
    ):
        prev_path_pipe = create_path_pipe(prev_path_pipe, map_pipe)

    logger.info("Marking rest of map for inside/outside")

    def mark_pipe(
        map_pipe: MapPipe, visited_recursive_coords: set[Coord2d] | None = None
    ) -> None:
        if map_pipe.pipe.inside is not Inside.Unknown:
            return

        if visited_recursive_coords is None:
            visited_recursive_coords = set()
        visited_recursive_coords.add(map_pipe.pipe.coord)

        for neighbor in map_pipe.neighbors.values():
            if neighbor.pipe.coord in visited_recursive_coords:
                continue
            mark_pipe(neighbor, visited_recursive_coords)
            if neighbor.pipe.inside in (Inside.Inside, Inside.Outside):
                map_pipe.pipe.inside = neighbor.pipe.inside
                return
        raise AssertionError()

    for _, pipe in map_data.iter_data():
        mark_pipe(pipe)

    def get_symbol_for_pipe(map_pipe: object) -> str:
        assert isinstance(map_pipe, MapPipe)
        if map_pipe.pipe.inside is Inside.Inside:
            return " "
        if map_pipe.pipe.inside is Inside.Outside:
            return "."
        if map_pipe.pipe.inside is Inside.InPath:
            return map_pipe.pipe.symbol
        return "#"

    if logger.isEnabledFor(logging.DEBUG):
        for map_line in map_data.str_lines(get_symbol_for_pipe):
            print(map_line)

    logger.info("Calculating inside locations")

    return sum(
        1 for _, pipe in map_data.iter_data() if pipe.pipe.inside is Inside.Inside
    )
