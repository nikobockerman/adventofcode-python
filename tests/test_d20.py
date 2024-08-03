from adventofcode import d20

_example1_input = r"""
broadcaster -> a, b, c
%a -> b
%b -> c
%c -> inv
&inv -> a
""".strip()

_example2_input = r"""
broadcaster -> a
%a -> inv, con
&inv -> b
%b -> con
&con -> output
""".strip()


def test_p1_example1():
    assert d20.p1(_example1_input) == 32000000


def test_p1_example2():
    assert d20.p1(_example2_input) == 11687500
