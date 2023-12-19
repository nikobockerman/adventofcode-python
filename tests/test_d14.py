from adventofcode import d14

example_input = """
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
    assert d14.p1(example_input) == 136


def test_p2():
    assert d14.p2(example_input) == 64
