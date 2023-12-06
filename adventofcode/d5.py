from dataclasses import dataclass, field
import logging
from typing import Iterable


@dataclass
class RangeMap:
    destination_start: int
    source_start: int
    length: int


@dataclass
class InputMaps:
    seed_to_soil_map: list[RangeMap] = field(default_factory=list, init=False)
    soil_to_fertilizer_map: list[RangeMap] = field(default_factory=list, init=False)
    fertilizer_to_water_map: list[RangeMap] = field(default_factory=list, init=False)
    water_to_light_map: list[RangeMap] = field(default_factory=list, init=False)
    light_to_temperature_map: list[RangeMap] = field(default_factory=list, init=False)
    temperature_to_humidity_map: list[RangeMap] = field(
        default_factory=list, init=False
    )
    humidity_to_location_map: list[RangeMap] = field(default_factory=list, init=False)


def _parse_input(lines: list[str]) -> tuple[list[int], InputMaps]:
    seeds = list(map(int, lines[0].strip().split(":")[1].strip().split()))

    maps = InputMaps()

    def save_mappings(mappings: list[RangeMap], mapping_name: str):
        prop_name = mapping_name.replace("-", "_").replace(" ", "_")
        setattr(maps, prop_name, mappings)

    def parse_mapping(line: str):
        return RangeMap(*map(int, line.strip().split()))

    mapping_name: str | None = None
    mappings: list[RangeMap] | None = None
    for line in lines[2:]:
        if not line:
            assert mappings is not None
            assert mapping_name is not None
            save_mappings(mappings, mapping_name)
            mapping_name = None
            mappings = None
            continue
        if line[-1] == ":":
            mapping_name = line[:-1]
            mappings = []
            continue

        assert mappings is not None
        mappings.append(parse_mapping(line))

    if mapping_name:
        assert mappings is not None
        save_mappings(mappings, mapping_name)

    return seeds, maps


def _get_location(maps: InputMaps, seed: int):
    def get_destination(mappings: list[RangeMap], source: int):
        for mapping in mappings:
            if source < mapping.source_start or source >= (
                mapping.source_start + mapping.length
            ):
                continue

            offset = source - mapping.source_start
            return mapping.destination_start + offset

        return source

    return get_destination(
        maps.humidity_to_location_map,
        get_destination(
            maps.temperature_to_humidity_map,
            get_destination(
                maps.light_to_temperature_map,
                get_destination(
                    maps.water_to_light_map,
                    get_destination(
                        maps.fertilizer_to_water_map,
                        get_destination(
                            maps.soil_to_fertilizer_map,
                            get_destination(maps.seed_to_soil_map, seed),
                        ),
                    ),
                ),
            ),
        ),
    )


def p1(input: str):
    seeds, maps = _parse_input(input.splitlines())
    logging.debug(f"{seeds=}")
    logging.debug(f"{maps=}")

    return min(map(lambda seed: _get_location(maps, seed), seeds))


@dataclass
class Range:
    start: int
    stop: int

    def __len__(self):
        return self.stop - self.start

    def partition(self, separator: "Range") -> tuple["Range", "Range", "Range"]:
        before = Range(0, 0)
        if self.start < separator.start:
            before = Range(self.start, min(self.stop, separator.start))

        overlap = Range(0, 0)
        if self.start < separator.stop and self.stop > separator.start:
            overlap = Range(
                max(self.start, separator.start), min(self.stop, separator.stop)
            )

        after = Range(0, 0)
        if self.stop > separator.stop:
            after = Range(max(self.start, separator.stop), self.stop)

        return before, overlap, after

    def overlaps(self, other: "Range") -> bool:
        _, overlap, _ = self.partition(other)
        return bool(overlap)


def _resolve_overlap(
    overlap_source: Range,
    dest_overlap: Range,
    next_ind: int,
    max_ind: int,
    mapping_ranges: list[list[tuple[Range, Range]]],
) -> Iterable[tuple[Range, Range]]:
    for resolved_source, top_level_source in _resolve_ranges(
        overlap_source, next_ind, max_ind, mapping_ranges
    ):
        resolved_offset_start = resolved_source.start - overlap_source.start
        resolved_offset_stop = overlap_source.stop - resolved_source.stop
        resolved_dest_start = dest_overlap.start + resolved_offset_start
        resolved_dest_stop = dest_overlap.stop - resolved_offset_stop
        assert resolved_dest_start < resolved_dest_stop

        yield Range(resolved_dest_start, resolved_dest_stop), top_level_source


def _resolve_ranges(
    dest_range_for_ind: Range,
    ind: int,
    max_ind: int,
    mapping_ranges: list[list[tuple[Range, Range]]],
    mapping_start_index: int | None = None,
) -> Iterable[tuple[Range, Range]]:
    assert ind <= max_ind

    if mapping_start_index is None:
        mapping_start_index = 0

    for mapping_range_ind, ranges in enumerate(
        mapping_ranges[ind], mapping_start_index
    ):
        mapping_dest_range, mapping_source_range = ranges

        (
            dest_before,
            dest_overlap,
            dest_after,
        ) = dest_range_for_ind.partition(mapping_dest_range)

        if not dest_overlap:
            assert bool(dest_before) != bool(dest_after)
            continue

        if dest_before:
            yield from _resolve_ranges(
                dest_before, ind, max_ind, mapping_ranges, mapping_range_ind + 1
            )

        overlap_offset_start = dest_overlap.start - mapping_dest_range.start
        overlap_offset_stop = mapping_dest_range.stop - dest_overlap.stop
        overlap_source_start = mapping_source_range.start + overlap_offset_start
        overlap_source_stop = mapping_source_range.stop - overlap_offset_stop
        assert overlap_source_start < overlap_source_stop

        overlap_source = Range(overlap_source_start, overlap_source_stop)

        if ind == max_ind:
            yield dest_overlap, overlap_source
        else:
            yield from _resolve_overlap(
                overlap_source, dest_overlap, ind + 1, max_ind, mapping_ranges
            )

        if dest_after:
            yield from _resolve_ranges(
                dest_after, ind, max_ind, mapping_ranges, mapping_range_ind + 1
            )

        return

    if ind == max_ind:
        yield dest_range_for_ind, dest_range_for_ind
    else:
        yield from _resolve_ranges(dest_range_for_ind, ind + 1, max_ind, mapping_ranges)


def p2(input: str):
    seed_data, maps = _parse_input(input.splitlines())

    seed_starts = seed_data[0::2]
    seed_lengths = seed_data[1::2]
    seed_ranges = [
        Range(start, start + length) for start, length in zip(seed_starts, seed_lengths)
    ]

    maps.humidity_to_location_map.sort(key=lambda map: map.destination_start)
    # maps.temperature_to_humidity_map.sort(key=lambda map: map.destination_start)
    # maps.light_to_temperature_map.sort(key=lambda map: map.destination_start)
    # maps.water_to_light_map.sort(key=lambda map: map.destination_start)
    # maps.fertilizer_to_water_map.sort(key=lambda map: map.destination_start)
    # maps.soil_to_fertilizer_map.sort(key=lambda map: map.destination_start)
    # maps.seed_to_soil_map.sort(key=lambda map: map.destination_start)

    mapping_ranges = [
        [
            (
                Range(
                    mapping.destination_start,
                    mapping.destination_start + mapping.length,
                ),
                Range(mapping.source_start, mapping.source_start + mapping.length),
            )
            for mapping in mappings
        ]
        for mappings in [
            maps.humidity_to_location_map,
            maps.temperature_to_humidity_map,
            maps.light_to_temperature_map,
            maps.water_to_light_map,
            maps.fertilizer_to_water_map,
            maps.soil_to_fertilizer_map,
            maps.seed_to_soil_map,
        ]
    ]
    max_ind = len(mapping_ranges) - 1

    location_dest_ranges = [dest_range for dest_range, _ in mapping_ranges[0]]
    if location_dest_ranges[0].start > 0:
        location_dest_ranges.insert(0, Range(0, location_dest_ranges[0].start))

    for initial_dest_range in location_dest_ranges:
        for resolved_location_range, resolved_seed_range in _resolve_ranges(
            initial_dest_range, 0, max_ind, mapping_ranges
        ):
            if any(
                resolved_seed_range.overlaps(seed_range) for seed_range in seed_ranges
            ):
                return resolved_location_range.start
    assert False
