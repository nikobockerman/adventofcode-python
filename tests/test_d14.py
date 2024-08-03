from adventofcode import d14

_example_input = """
O....#....
O.OO#....#
.....##...
OO.#O....O
.O.....O#.
O.#..O.#.#
..O..#O..O
.......O..
#....###..
#OO..#....
""".strip()


def test_p1():
    assert d14.p1(_example_input) == 136


def test_p2():
    assert d14.p2(_example_input) == 64
