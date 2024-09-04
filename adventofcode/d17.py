import heapq
import logging
from dataclasses import dataclass, field

from adventofcode.tooling.coordinates import Coord2d, X, Y
from adventofcode.tooling.directions import CardinalDirection as Dir
from adventofcode.tooling.map import Map2d

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True, order=True)
class _PathPosition:
    coord: Coord2d = field(compare=False)
    direction: Dir = field(compare=False)
    next_coord: Coord2d = field(compare=False)
    moves_straight: int = field(compare=False)
    heat_loss: int = field(compare=False)
    estimated_total_heat_loss: int


def _estimate_remaining_heat_loss(start: Coord2d, destination: Coord2d) -> int:
    return start.distance_to_int(destination)


class _PriorityQueue:
    __slots__ = ("_queue",)

    def __init__(self) -> None:
        self._queue = list[_PathPosition]()

    def put(self, item: _PathPosition) -> None:
        heapq.heappush(self._queue, item)

    def pop(self) -> _PathPosition:
        return heapq.heappop(self._queue)


def _create_prio_queue(start_pos: Coord2d, destination: Coord2d) -> _PriorityQueue:
    queue = _PriorityQueue()
    for start_dir in (Dir.S, Dir.E):
        new_next_coord = start_pos.adjoin(start_dir)
        pos = _PathPosition(
            start_pos,
            start_dir,
            new_next_coord,
            1,
            0,
            _estimate_remaining_heat_loss(new_next_coord, destination),
        )
        _logger.debug("Adding %s to queue", pos)
        queue.put(pos)
    return queue


type _MinCacheValidValue = list[tuple[int, int]]
type _MinCacheTooFewValue = dict[int, int]
# type _VisitedMinCacheValue = tuple[
#    _VisitedMinCacheValidValue, _VisitedMinCacheTooFewValue
# ]
# type _VisitedMinCache = dict[Y, dict[X, dict[Dir, _VisitedMinCacheValue]]]

# type _MinCacheValid = dict[Y, dict[X, dict[Dir, _MinCacheValidValue]]]
# type _MinCacheTooFew = dict[Y, dict[X, dict[Dir, _MinCacheTooFewValue]]]
type _MinCacheValid = dict[tuple[Y, X, Dir], _MinCacheValidValue]
type _MinCacheTooFew = dict[tuple[Y, X, Dir], _MinCacheTooFewValue]


@dataclass(slots=True)
class _ResolutionData:
    min_straight_moves: int
    max_straight_moves: int
    map_: Map2d[int]
    min_cache_valid: _MinCacheValid = field(default_factory=dict)
    min_cache_too_few: _MinCacheTooFew = field(default_factory=dict)


def _get_or_create_cache_value_y_x_dir[T](
    cache: dict[Y, dict[X, dict[Dir, T]]], y: Y, x: X, dir_: Dir
) -> tuple[dict[Dir, T], T | None]:
    y_value = cache.get(y)
    if y_value is None:
        y_value = {}
        cache[y] = y_value

    x_value = y_value.get(x)
    if x_value is None:
        x_value = {}
        y_value[x] = x_value

    return x_value, x_value.get(dir_)


def _get_or_create_cache_value_valid(
    cache: _MinCacheValid, y: Y, x: X, dir_: Dir
) -> _MinCacheValidValue:
    key = (y, x, dir_)
    value = cache.get(key)
    if value is None:
        value = []
        cache[key] = value

    return value
    # x_value, dir_value = _get_or_create_cache_value_y_x_dir(cache, y, x, dir_)
    # if dir_value is None:
    #    dir_value = []
    #    x_value[dir_] = dir_value

    # return dir_value


def _get_or_create_cache_value_too_few(
    cache: _MinCacheTooFew, y: Y, x: X, dir_: Dir
) -> _MinCacheTooFewValue:
    key = (y, x, dir_)
    value = cache.get(key)
    if value is None:
        value = {}
        cache[key] = value

    return value
    # x_value, dir_value = _get_or_create_cache_value_y_x_dir(cache, y, x, dir_)
    # if dir_value is None:
    #    dir_value = {}
    #    x_value[dir_] = dir_value
    # return dir_value


def _cache_update(
    coord: Coord2d,
    dir_: Dir,
    moves_straight: int,
    heat_loss: int,
    data: _ResolutionData,
) -> bool:
    if moves_straight >= data.min_straight_moves:
        valid_cache = _get_or_create_cache_value_valid(
            data.min_cache_valid, coord.y, coord.x, dir_
        )
        cached_min = next(
            (
                heat_loss
                for straight_so_far, heat_loss in valid_cache
                if moves_straight >= straight_so_far
            ),
            None,
        )
        if cached_min is not None and cached_min <= heat_loss:
            # _logger.debug("Already seen with better or equal heat loss: %d", cached_min)
            return False
        valid_cache.append((moves_straight, heat_loss))
        valid_cache.sort(key=lambda x: x[1])
        # _logger.debug(
        #    "Cached before but with worse heat loss. New entries: %s", valid_cache
        # )
    else:
        too_few_cache = _get_or_create_cache_value_too_few(
            data.min_cache_too_few, coord.y, coord.x, dir_
        )
        cached_min = too_few_cache.get(moves_straight)
        if cached_min is not None and cached_min <= heat_loss:
            # _logger.debug("Already seen with better or equal heat loss: %d", cached_min)
            return False
        too_few_cache[moves_straight] = heat_loss
        # _logger.debug(
        #    "Cached before but with worse heat loss. New entries: %s",
        #    too_few_moves_cache,
        # )
    return True


def _get_next_position_in_direction(
    pos: _PathPosition,
    new_dir: Dir,
    new_heat_loss: int,
    destination: Coord2d,
    data: _ResolutionData,
) -> _PathPosition | None:
    if new_dir != pos.direction and pos.moves_straight < data.min_straight_moves:
        return None
    if new_dir == pos.direction:
        if pos.moves_straight >= data.max_straight_moves:
            return None
        new_moves_straight = pos.moves_straight + 1
    else:
        new_moves_straight = 1

    _logger.debug("Processing move %s -> %s", pos.next_coord, new_dir)

    if not _cache_update(
        pos.next_coord, new_dir, new_moves_straight, new_heat_loss, data
    ):
        return None
    # cached_pos = data.visited_min_cache.get((pos.next_coord, new_dir))
    # if cached_pos is None:
    #    if new_moves_straight >= data.min_straight_moves:
    #        data.visited_min_cache[(pos.next_coord, new_dir)] = (
    #            [(new_moves_straight, new_heat_loss)],
    #            [],
    #        )
    #    else:
    #        data.visited_min_cache[(pos.next_coord, new_dir)] = (
    #            [],
    #            [(new_moves_straight, new_heat_loss)],
    #        )
    # else:
    #    if new_moves_straight >= data.min_straight_moves:
    #        cache_list = cached_pos[0]
    #        cached_min = next(
    #            (
    #                heat_loss
    #                for straight_so_far, heat_loss in cache_list
    #                if new_moves_straight >= straight_so_far
    #            ),
    #            None,
    #        )
    #    else:
    #        cache_list = cached_pos[1]
    #        cached_min = next(
    #            (
    #                heat_loss
    #                for straight_so_far, heat_loss in cache_list
    #                if new_moves_straight == straight_so_far
    #            ),
    #            None,
    #        )
    #    if cached_min is not None and cached_min <= new_heat_loss:
    #        _logger.debug(
    #            "Already seen with better or equal heat loss: %d",
    #            cached_min,
    #        )
    #        return None
    #    cache_list.append((new_moves_straight, new_heat_loss))
    #    cache_list.sort(key=lambda x: x[1])
    #    _logger.debug(
    #        "Cached before but with worse heat loss. New entries: %s",
    #        cached_pos,
    #    )

    new_next_coord = pos.next_coord.adjoin(new_dir)

    if (
        new_next_coord.x < data.map_.tl_x
        or new_next_coord.x > data.map_.br_x
        or new_next_coord.y < data.map_.tl_y
        or new_next_coord.y > data.map_.br_y
    ):
        _logger.debug("Outside of map")
        return None

    return _PathPosition(
        pos.next_coord,
        new_dir,
        new_next_coord,
        new_moves_straight,
        new_heat_loss,
        new_heat_loss + _estimate_remaining_heat_loss(new_next_coord, destination),
    )


def _resolve(input_str: str, min_straight_moves: int, max_straight_moves: int) -> int:
    map_ = Map2d((int(c) for c in line) for line in input_str.splitlines())
    start_pos = Coord2d(map_.tl_y, map_.tl_x)
    destination = Coord2d(map_.br_y, map_.br_x)

    queue = _create_prio_queue(start_pos, destination)

    data = _ResolutionData(min_straight_moves, max_straight_moves, map_)

    result: int | None = None
    while True:
        pos = queue.pop()

        _logger.debug("Processing cheapest: %s", pos)

        new_heat_loss = pos.heat_loss + map_.get(pos.next_coord.y, pos.next_coord.x)

        if pos.next_coord == destination:
            if result is None or new_heat_loss < result:
                _logger.debug(
                    "Found new shortest path: new_heat_loss=%s", new_heat_loss
                )
                result = new_heat_loss
            elif new_heat_loss > result:
                break
            continue

        for new_dir in (
            pos.direction,
            pos.direction.rotate_counterclockwise(),
            pos.direction.rotate_clockwise(),
        ):
            new_pos = _get_next_position_in_direction(
                pos, new_dir, new_heat_loss, destination, data
            )
            if new_pos is None:
                continue

            _logger.debug("Adding %s to queue", new_pos)
            queue.put(new_pos)

    assert result is not None
    return result


def p1(input_str: str) -> int:
    return _resolve(input_str, 0, 3)


def p2(input_str: str) -> int:
    return _resolve(input_str, 4, 10)


if __name__ == "__main__":
    import cProfile
    import pathlib
    from argparse import ArgumentParser
    from typing import Literal, assert_never

    import pyinstrument

    parser = ArgumentParser()
    parser.add_argument("profiler", type=str, choices=["cProfile", "pyinstrument"])
    args = parser.parse_args()
    profiler: Literal["cProfile", "pyinstrument"]
    profiler = args.profiler

    input_str = (pathlib.Path(__file__).parent / "input-d17.txt").read_text().strip()

    if profiler == "cProfile":
        with cProfile.Profile() as pr:
            result = p2(input_str)
            pr.print_stats(sort="cumtime")
    elif profiler == "pyinstrument":
        with pyinstrument.profile():
            result = p2(input_str)
    else:
        assert_never(profiler)
    print(result)
