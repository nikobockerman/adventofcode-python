from adventofcode import d13

_example_input = """
#.##..##.
..#.##.#.
##......#
##......#
..#.##.#.
..##..##.
#.#.##.#.

#...##..#
#....#..#
..##..###
#####.##.
#####.##.
..##..###
#....#..#
""".strip()


def test_p1() -> None:
    assert d13.p1(_example_input) == 405


def test_p2() -> None:
    assert d13.p2(_example_input) == 400
