from adventofcode import d11

_example_input = """
...#......
.......#..
#.........
..........
......#...
.#........
.........#
..........
.......#..
#...#.....
""".strip()


def test_p1_example_1():
    assert d11.p1(_example_input) == 374


def test_p2_example_10() -> None:
    assert d11.calculate_distance_between_galaxies(_example_input, 10) == 1_030
