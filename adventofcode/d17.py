from __future__ import annotations

import heapq
import logging
import pathlib
from dataclasses import InitVar, dataclass, field
from typing import Iterator, Literal, NewType, assert_never

from adventofcode.tooling.coordinates import (
    Coord2d,
    X,
    Y,
    adjoin_east,
    adjoin_north,
    adjoin_south,
    adjoin_west,
)
from adventofcode.tooling.directions import CardinalDirection as Dir
from adventofcode.tooling.map import Map2d

_logger = logging.getLogger(__name__)

# type HeatLoss = int
HeatLoss = NewType("HeatLoss", float)

## Optimization TODO
# .. Compare performance with timeit of hash(Coord2d) vs hash(tuple[int, int]) -> tuple is significantly faster
# .. Monitor the size of queue and count after each result finding -> could removing too long entries from queue be useful? -> No
# 3. Remove too_few cache and jumping ahead to the next possible straight step after turn
# 4. Change A* to estimate remaining heat loss based on average heat loss per square so far
# 5. Remove min_cache and rely on the A*


@dataclass(slots=True)
class _PathPosition:
    reached_coord: Coord2d
    direction_when_reached: Dir
    moves_straight: int
    # distance_to_reach: int
    heat_loss_to_reach: HeatLoss
    estimated_total_heat_loss: HeatLoss

    # prev_position: _PathPosition | None = field(repr=False)

    # map_: InitVar[Map2d[int]]
    # prev_coord: InitVar[Coord2d]
    # heat_loss_to_reach_prev: InitVar[HeatLoss]
    # distance_to_reach_prev: InitVar[int]
    # destination: InitVar[Coord2d]

    # distance_to_reach: int = field(init=False)
    # heat_loss_to_reach: HeatLoss = field(init=False)
    # estimated_total_heat_loss: HeatLoss = field(init=False)

    """
    def __post_init__(
        self,
        map_: Map2d[int],
        prev_coord: Coord2d,
        heat_loss_to_reach_prev: HeatLoss,
        distance_to_reach_prev: int,
        destination: Coord2d,
    ) -> None:
        self.distance_to_reach = distance_to_reach_prev + prev_coord.distance_to_int(
            self.reached_coord
        )
        move_dir = prev_coord.dir_to(self.reached_coord)
        first_position_in_route = prev_coord.adjoin(move_dir)

        if first_position_in_route == self.reached_coord:
            heat_loss_on_route = map_.get(self.reached_coord.y, self.reached_coord.x)
        else:
            heat_loss_on_route = sum(
                heat_loss
                for _, x_and_heat_loss in map_.iter_data_by_lines(
                    first_position_in_route.y,
                    first_position_in_route.x,
                    self.reached_coord.y,
                    self.reached_coord.x,
                )
                for _, heat_loss in x_and_heat_loss
            )

        self.heat_loss_to_reach = HeatLoss(heat_loss_to_reach_prev + heat_loss_on_route)
        average_heat_loss_to_reach = self.heat_loss_to_reach / self.distance_to_reach
        self.estimated_total_heat_loss = HeatLoss(
            self.heat_loss_to_reach
            + self.reached_coord.distance_to_int(destination)
            * average_heat_loss_to_reach
        )
    """

    def __lt__(self, other: _PathPosition) -> bool:
        return self.estimated_total_heat_loss < other.estimated_total_heat_loss


def _get_heat_loss_in_route(
    start_y: Y, start_x: X, last_y: Y, last_x: X, dir_: Dir, map_: Map2d[int]
) -> HeatLoss:
    # if start_y == last_y and start_x == last_x:
    #    return

    if dir_ == Dir.S:
        adjoin_func = adjoin_south
    elif dir_ == Dir.N:
        adjoin_func = adjoin_north
    else:
        # assert start_y == last_y
        if dir_ == Dir.E:
            return HeatLoss(sum(map_._sequence_data[start_y][start_x + 1 : last_x + 1]))
        if dir_ == Dir.W:
            return HeatLoss(sum(map_._sequence_data[start_y][last_x:start_x]))
        assert_never(dir_)

    # assert start_x == last_x

    cur_y, cur_x = adjoin_func(start_y, start_x)
    heat_loss: HeatLoss = HeatLoss(map_.get(cur_y, cur_x))
    while cur_y != last_y or cur_x != last_x:
        cur_y, cur_x = adjoin_func(cur_y, cur_x)
        heat_loss = HeatLoss(heat_loss + map_.get(cur_y, cur_x))
    return heat_loss


def _calculate_heat_loss_to_reach(
    reached_coord: Coord2d,
    prev_coord: Coord2d,
    prev_heat_loss_to_reach: HeatLoss,
    distance_from_prev: int,
    direction_from_prev: Dir,
    map_: Map2d[int],
) -> HeatLoss:
    assert distance_from_prev >= 1
    if distance_from_prev == 1:
        heat_loss_on_route = map_.get(reached_coord.y, reached_coord.x)
    else:
        heat_loss_on_route = _get_heat_loss_in_route(
            prev_coord.y,
            prev_coord.x,
            reached_coord.y,
            reached_coord.x,
            direction_from_prev,
            map_,
        )

    return HeatLoss(prev_heat_loss_to_reach + heat_loss_on_route)


def _estimate_remaining_heat_loss(start: Coord2d, destination: Coord2d) -> HeatLoss:
    # return HeatLoss(start.distance_to_int(destination))
    return HeatLoss(start.distance_to(destination))


class _PriorityQueue:
    __slots__ = ("_queue", "max_size")

    def __init__(self) -> None:
        self._queue = list[_PathPosition]()
        self.max_size = 0

    def put(self, item: _PathPosition) -> None:
        heapq.heappush(self._queue, item)
        size = len(self._queue)
        if size > self.max_size:
            self.max_size = size

    def pop(self) -> _PathPosition:
        return heapq.heappop(self._queue)

    def putpop(self, item: _PathPosition) -> _PathPosition:
        result = heapq.heappushpop(self._queue, item)
        size = len(self._queue)
        if size > self.max_size:
            self.max_size = size
        return result


def _create_prio_queue(start_pos: Coord2d, data: _ResolutionData) -> _PriorityQueue:
    queue = _PriorityQueue()
    for start_dir in (Dir.S, Dir.E):
        moves_straight = data.min_straight_moves
        reached_coord = start_pos.get_relative(start_dir, data.min_straight_moves)
        heat_loss = _calculate_heat_loss_to_reach(
            reached_coord, start_pos, HeatLoss(0), moves_straight, start_dir, data.map_
        )
        estimated_total_heat_loss = HeatLoss(
            heat_loss + _estimate_remaining_heat_loss(reached_coord, data.destination)
        )
        pos = _PathPosition(
            reached_coord,
            start_dir,
            moves_straight,
            heat_loss,
            estimated_total_heat_loss,
        )
        _logger.debug("Adding %s to queue", pos)
        queue.put(pos)
    return queue


type _MinCacheValidValue = list[tuple[int, HeatLoss]]
type _MinCacheValid = dict[tuple[Y, X, Dir], _MinCacheValidValue]


@dataclass(slots=True)
class _ResolutionData:
    min_straight_moves: int
    max_straight_moves: int
    map_: Map2d[int]
    destination: Coord2d
    min_cache_valid: _MinCacheValid = field(default_factory=dict)


def _get_or_create_cache_value_valid(
    cache: _MinCacheValid, y: Y, x: X, dir_: Dir
) -> _MinCacheValidValue:
    key = (y, x, dir_)
    value = cache.get(key)
    if value is None:
        value = []
        cache[key] = value

    return value


def _valid_cache_check_and_update_for_min_heat_loss(
    coord: Coord2d,
    dir_: Dir,
    moves_straight: int,
    heat_loss: HeatLoss,
    cache: _MinCacheValid,
) -> bool:
    valid_cache = _get_or_create_cache_value_valid(cache, coord.y, coord.x, dir_)
    cached_min = next(
        (
            heat_loss
            for straight_so_far, heat_loss in valid_cache
            if moves_straight >= straight_so_far
        ),
        None,
    )
    if cached_min is not None and cached_min <= heat_loss:
        _logger.debug("Already seen with better or equal heat loss: %d", cached_min)
        return False
    valid_cache.append((moves_straight, heat_loss))
    valid_cache.sort(key=lambda x: x[1])
    if cached_min is not None:
        _logger.debug(
            "Cached before but with worse heat loss. New entries: %s", valid_cache
        )
    else:
        _logger.debug("Not cached before. New entries: %s", valid_cache)
    return True


def _get_next_position_straight_ahead(
    pos: _PathPosition, data: _ResolutionData
) -> _PathPosition | None:
    assert pos.moves_straight >= data.min_straight_moves
    distance_from_prev = 1
    moves_straight = pos.moves_straight + distance_from_prev
    if moves_straight > data.max_straight_moves:
        return None

    direction = pos.direction_when_reached

    reached_coord = pos.reached_coord.adjoin(direction)

    if not data.map_.contains(reached_coord.y, reached_coord.x):
        _logger.debug("Outside of map")
        return None

    heat_loss = _calculate_heat_loss_to_reach(
        reached_coord,
        pos.reached_coord,
        pos.heat_loss_to_reach,
        distance_from_prev,
        direction,
        data.map_,
    )

    _logger.debug(
        "Processing straight move %s -> %s: moves_straight=%d, heat_loss=%f",
        direction,
        reached_coord,
        moves_straight,
        heat_loss,
    )

    if not _valid_cache_check_and_update_for_min_heat_loss(
        reached_coord, direction, moves_straight, heat_loss, data.min_cache_valid
    ):
        return None

    estimated_total_heat_loss = HeatLoss(
        heat_loss + _estimate_remaining_heat_loss(reached_coord, data.destination)
    )

    return _PathPosition(
        reached_coord,
        direction,
        moves_straight,
        heat_loss,
        estimated_total_heat_loss,
    )


def _get_next_position_after_turn(
    pos: _PathPosition, direction: Dir, data: _ResolutionData
) -> _PathPosition | None:
    moves_straight = data.min_straight_moves

    reached_coord = pos.reached_coord.get_relative(direction, moves_straight)

    if not data.map_.contains(reached_coord.y, reached_coord.x):
        # _logger.debug("Outside of map")
        return None

    heat_loss = _calculate_heat_loss_to_reach(
        reached_coord,
        pos.reached_coord,
        pos.heat_loss_to_reach,
        moves_straight,
        direction,
        data.map_,
    )

    _logger.debug(
        "Processing turn move %s -> %s: moves_straight=%d, heat_loss=%f",
        direction,
        reached_coord,
        moves_straight,
        heat_loss,
    )

    if not _valid_cache_check_and_update_for_min_heat_loss(
        reached_coord, direction, moves_straight, heat_loss, data.min_cache_valid
    ):
        return None

    estimated_total_heat_loss = HeatLoss(
        heat_loss + _estimate_remaining_heat_loss(reached_coord, data.destination)
    )

    return _PathPosition(
        reached_coord,
        direction,
        moves_straight,
        heat_loss,
        estimated_total_heat_loss,
    )


def _map_route(pos: _PathPosition, map_: Map2d[int]) -> Iterator[str]:
    route: list[Coord2d] = []
    p = pos
    sum_ = 0
    while p is not None:
        route.append(p.reached_coord)
        sum_ += map_.get(p.reached_coord.y, p.reached_coord.x)
        p = p.prev_position
    route.append(Coord2d(Y(0), X(0)))
    route = list(reversed(route))

    min_x = min(x.x for x in route)
    min_y = min(x.y for x in route)
    max_x = max(x.x for x in route)
    max_y = max(x.y for x in route)
    yield (f"Count: {len(route)}. Area(x: {min_x} - {max_x}, y: {min_y} - {max_y})")
    yield f"Total heat loss: {sum_}"

    for y in range(min_y, max_y + 1):
        row = ""
        for x in range(min_x, max_x + 1):
            if Coord2d(Y(y), X(x)) in route:
                v = map_.get(Y(y), X(x))
            else:
                v = "."
            row += f"{v}"
        yield row


def _resolve(
    input_str: str, min_straight_moves: int, max_straight_moves: int
) -> HeatLoss:
    map_ = Map2d((int(c) for c in line) for line in input_str.splitlines())
    start_pos = Coord2d(map_.tl_y, map_.tl_x)
    destination = Coord2d(map_.br_y, map_.br_x)
    data = _ResolutionData(min_straight_moves, max_straight_moves, map_, destination)

    queue = _create_prio_queue(start_pos, data)

    result: HeatLoss | None = None
    pos = queue.pop()
    count = 0
    while True:
        count += 1
        _logger.debug("Processing cheapest: %s", pos)

        if pos.reached_coord == destination:
            if result is None or pos.heat_loss_to_reach < result:
                _logger.debug(
                    "Found new shortest path: new_heat_loss=%s", pos.heat_loss_to_reach
                )
                _logger.info("Max priority queue size: %d", queue.max_size)
                _logger.info("Count: %d", count)
                result = pos.heat_loss_to_reach
                # _logger.debug(
                #    "Result route: \n%s",
                #    "\n".join(_map_route(pos, map_)),
                # )
            elif pos.heat_loss_to_reach > result:
                break

            pos = queue.pop()
            continue

        new_positions: list[_PathPosition] = []
        new_pos = _get_next_position_straight_ahead(pos, data)
        if new_pos is not None:
            _logger.debug("Adding %s to queue", new_pos)
            new_positions.append(new_pos)

        for new_dir in (
            pos.direction_when_reached.rotate_counterclockwise(),
            pos.direction_when_reached.rotate_clockwise(),
        ):
            new_pos = _get_next_position_after_turn(pos, new_dir, data)
            if new_pos is None:
                continue

            # queue.put(new_pos)
            _logger.debug("Adding %s to queue", new_pos)
            new_positions.append(new_pos)

        if not new_positions:
            pos = queue.pop()
        else:
            for new_position in new_positions[:-1]:
                # _logger.debug("Adding %s to queue", new_position)
                queue.put(new_position)
            # _logger.debug("Adding %s to queue", new_positions[-1])
            pos = queue.putpop(new_positions[-1])

    assert result is not None
    _logger.info("Max priority queue size: %d", queue.max_size)
    _logger.info("Count: %d", count)
    return result


def p1(input_str: str) -> int:
    result = _resolve(input_str, 1, 3)
    assert result.is_integer()
    return int(result)


def p2(input_str: str) -> int:
    result = _resolve(input_str, 4, 10)
    assert result.is_integer()
    return int(result)


def main() -> None:
    import argparse
    import cProfile
    import pstats

    import pyinstrument

    parser = argparse.ArgumentParser()
    parser.add_argument("profiler", type=str, choices=["cProfile", "pyinstrument"])
    args = parser.parse_args()
    profiler: Literal["cProfile", "pyinstrument"]
    profiler = args.profiler

    input_str = (pathlib.Path(__file__).parent / "input-d17.txt").read_text().strip()

    if profiler == "cProfile":
        with cProfile.Profile() as pr:
            result = p2(input_str)
            stats = pstats.Stats(pr).strip_dirs()
            stats.sort_stats("tottime").print_stats(0.2)
            stats.sort_stats("cumtime").print_stats(0.2)
    elif profiler == "pyinstrument":
        with pyinstrument.profile():
            result = p2(input_str)
    else:
        assert_never(profiler)
    print(result)


if __name__ == "__main__":
    main()
