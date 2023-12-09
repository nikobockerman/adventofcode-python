from adventofcode import d8

example_input_p1 = """
LLR

AAA = (BBB, BBB)
BBB = (AAA, ZZZ)
ZZZ = (ZZZ, ZZZ)
""".strip()


def test_d8_p1() -> None:
    assert d8.p1(example_input_p1) == 6


example_input_p2 = """
LR

11A = (11B, XXX)
11B = (XXX, 11Z)
11Z = (11B, XXX)
22A = (22B, XXX)
22B = (22C, 22C)
22C = (22Z, 22Z)
22Z = (22B, 22B)
XXX = (XXX, XXX)
""".strip()


def test_d8_p2() -> None:
    assert d8.p2(example_input_p2) == 6
