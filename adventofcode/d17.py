import heapq
import logging
from dataclasses import dataclass, field

from adventofcode.tooling.directions import CardinalDirection as Dir
from adventofcode.tooling.map import Coord2d, Map2d

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
    def __init__(self) -> None:
        self._queue = list[_PathPosition]()

    def put(self, item: _PathPosition) -> None:
        heapq.heappush(self._queue, item)

    def pop(self) -> _PathPosition:
        return heapq.heappop(self._queue)


def _resolve(input_str: str, min_straight_moves: int, max_straight_moves: int) -> int:
    map_ = Map2d((int(c) for c in line) for line in input_str.splitlines())
    queue = _PriorityQueue()
    start_pos = Coord2d(map_.first_x, map_.first_y)
    destination = Coord2d(map_.last_x, map_.last_y)
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

    visited_min_cache: dict[
        tuple[Coord2d, Dir], tuple[list[tuple[int, int]], list[tuple[int, int]]]
    ] = {}

    result: int | None = None
    while True:
        pos = queue.pop()

        _logger.debug("Processing cheapest: %s", pos)

        new_heat_loss = pos.heat_loss + map_.get(pos.next_coord)

        if pos.next_coord == destination:
            if result is None or new_heat_loss < result:
                _logger.debug(f"Found new shortest path: {new_heat_loss=}")
                result = new_heat_loss
            elif new_heat_loss > result:
                break
            continue

        for new_dir in (
            pos.direction,
            pos.direction.rotate_counterclockwise(),
            pos.direction.rotate_clockwise(),
        ):
            if new_dir != pos.direction and pos.moves_straight < min_straight_moves:
                continue
            if new_dir == pos.direction:
                if pos.moves_straight >= max_straight_moves:
                    continue
                new_moves_straight = pos.moves_straight + 1
            else:
                new_moves_straight = 1

            _logger.debug("Processing move %s -> %s", pos.next_coord, new_dir)

            cached_pos = visited_min_cache.get((pos.next_coord, new_dir))
            if cached_pos is None:
                if new_moves_straight >= min_straight_moves:
                    visited_min_cache[(pos.next_coord, new_dir)] = (
                        [(new_moves_straight, new_heat_loss)],
                        [],
                    )
                else:
                    visited_min_cache[(pos.next_coord, new_dir)] = (
                        [],
                        [(new_moves_straight, new_heat_loss)],
                    )
            else:
                if new_moves_straight >= min_straight_moves:
                    cache_list = cached_pos[0]
                    cached_min = next(
                        (
                            heat_loss
                            for straight_so_far, heat_loss in cache_list
                            if new_moves_straight >= straight_so_far
                        ),
                        None,
                    )
                else:
                    cache_list = cached_pos[1]
                    cached_min = next(
                        (
                            heat_loss
                            for straight_so_far, heat_loss in cache_list
                            if new_moves_straight == straight_so_far
                        ),
                        None,
                    )
                if cached_min is not None and cached_min <= new_heat_loss:
                    _logger.debug(
                        "Already seen with better or equal heat loss: %d",
                        cached_min,
                    )
                    continue
                cache_list.append((new_moves_straight, new_heat_loss))
                cache_list.sort(key=lambda x: x[1])
                _logger.debug(
                    "Cached before but with worse heat loss. New entries: %s",
                    cached_pos,
                )

            new_next_coord = pos.next_coord.adjoin(new_dir)

            if (
                new_next_coord.x < map_.first_x
                or new_next_coord.x > map_.last_x
                or new_next_coord.y < map_.first_y
                or new_next_coord.y > map_.last_y
            ):
                _logger.debug("Outside of map")
                continue

            new_pos = _PathPosition(
                pos.next_coord,
                new_dir,
                new_next_coord,
                new_moves_straight,
                new_heat_loss,
                new_heat_loss
                + _estimate_remaining_heat_loss(new_next_coord, destination),
            )
            _logger.debug("Adding %s to queue", new_pos)
            queue.put(new_pos)

    assert result is not None
    return result


def p1(input_str: str) -> int:
    return _resolve(input_str, 0, 3)


def p2(input_str: str) -> int:
    return _resolve(input_str, 4, 10)
