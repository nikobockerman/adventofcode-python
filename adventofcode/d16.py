import logging

from adventofcode.tooling.map import Coord2d, Dir, Map2d

logger = logging.getLogger(__name__)


class _SplitterExitCache:
    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: dict[
            tuple[Coord2d, Dir], tuple[set[Coord2d], list[tuple[Coord2d, Dir]]]
        ] = {}

    def get(
        self, coord: Coord2d, dir_: Dir
    ) -> tuple[set[Coord2d], list[tuple[Coord2d, Dir]]] | None:
        return self._cache.get((coord, dir_))

    def add(
        self,
        coord: Coord2d,
        dir_: Dir,
        visited: set[Coord2d],
        next_splitter_exits: list[tuple[Coord2d, Dir]],
    ) -> None:
        assert (coord, dir_) not in self._cache
        self._cache[(coord, dir_)] = (visited, next_splitter_exits)


def _process_splitter_exit(
    coord: Coord2d, dir_: Dir, map_: Map2d[str]
) -> tuple[set[Coord2d], list[tuple[Coord2d, Dir]]]:
    visited: set[Coord2d] = set()
    next_splitter_exits: list[tuple[Coord2d, Dir]] = []
    while True:
        if (
            coord.x < map_.first_x
            or coord.x > map_.last_x
            or coord.y < map_.first_y
            or coord.y > map_.last_y
        ):
            logger.debug("Out of map")
            break
        visited.add(coord)

        symbol = map_.get(coord)
        logger.debug("Coord: %s Dir: %s Symbol: %s", coord, dir_, symbol)

        if symbol == ".":
            coord = coord.adjoin(dir_)
            continue

        if symbol == "/":
            dir_ = dir_.rotate_right() if dir_ in (Dir.N, Dir.S) else dir_.rotate_left()
            coord = coord.adjoin(dir_)
            continue

        if symbol == "\\":
            dir_ = dir_.rotate_left() if dir_ in (Dir.N, Dir.S) else dir_.rotate_right()
            coord = coord.adjoin(dir_)
            continue

        if symbol == "-":
            if dir_ in (Dir.N, Dir.S):
                next_splitter_exits.extend(
                    [(coord.adjoin(Dir.E), Dir.E), (coord.adjoin(Dir.W), Dir.W)]
                )
            else:
                next_splitter_exits.append((coord.adjoin(dir_), dir_))
            break

        if symbol == "|":
            if dir_ in (Dir.E, Dir.W):
                next_splitter_exits.extend(
                    [(coord.adjoin(Dir.N), Dir.N), (coord.adjoin(Dir.S), Dir.S)]
                )
            else:
                next_splitter_exits.append((coord.adjoin(dir_), dir_))
            break
    return visited, next_splitter_exits


def _try_one_enter(
    enter_coord: Coord2d,
    enter_dir: Dir,
    map_: Map2d[str],
    exit_cache: _SplitterExitCache,
) -> int:
    visited: set[Coord2d] = set()
    processed_splitter_exits: set[tuple[Coord2d, Dir]] = set()
    processing_queue: list[tuple[Coord2d, Dir]] = [(enter_coord, enter_dir)]

    while processing_queue:
        coord, dir_ = processing_queue.pop(0)
        logger.debug("Start processing %s -> %s", coord, dir_)
        if (coord, dir_) in processed_splitter_exits:
            logger.debug("Already processed")
            continue
        processed_splitter_exits.add((coord, dir_))
        logger.debug("Visited count so far: %d", len(visited))

        cached = exit_cache.get(coord, dir_)
        if cached is not None:
            logger.debug("Cache hit")
            processed_visited, next_splitter_exits = cached
        else:
            processed_visited, next_splitter_exits = _process_splitter_exit(
                coord, dir_, map_
            )
            exit_cache.add(coord, dir_, processed_visited, next_splitter_exits)
        visited |= processed_visited
        processing_queue.extend(next_splitter_exits)
    return len(visited)


def p1(input_str: str) -> int:
    map_ = Map2d(input_str.splitlines())
    exit_cache = _SplitterExitCache()
    return _try_one_enter(Coord2d(0, 0), Dir.E, map_, exit_cache)


def p2(input_str: str) -> int:
    map_ = Map2d(input_str.splitlines())
    exit_cache = _SplitterExitCache()
    results: list[tuple[Coord2d, Dir, int]] = []

    for x in range(map_.first_x, map_.last_x + 1):
        for y in (map_.first_y, map_.last_y):
            coord = Coord2d(x, y)
            dir_ = Dir.S if y == 0 else Dir.N
            logging.info("Trying %s -> %s", coord, dir_)
            result = _try_one_enter(coord, dir_, map_, exit_cache)
            logging.info("Result %s -> %s = %d", coord, dir_, result)
            results.append((coord, dir_, result))
    for y in range(map_.first_y, map_.last_y + 1):
        for x in (map_.first_x, map_.last_x):
            coord = Coord2d(x, y)
            dir_ = Dir.E if x == 0 else Dir.W
            logging.info("Trying %s -> %s", coord, dir_)
            result = _try_one_enter(coord, dir_, map_, exit_cache)
            logging.info("Result %s -> %s = %d", coord, dir_, result)
            results.append((coord, dir_, result))

    results.sort(key=lambda r: r[2])
    return results[-1][2]
