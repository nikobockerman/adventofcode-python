from adventofcode import d20

example1_input = r"""
broadcaster -> a, b, c
%a -> b
%b -> c
%c -> inv
&inv -> a
""".strip()

example2_input = r"""
broadcaster -> a
%a -> inv, con
&inv -> b
%b -> con
&con -> output
""".strip()


def test_p1_example1():
    assert d20.p1(example1_input) == 32000000


def test_p1_example2():
    assert d20.p1(example2_input) == 11687500


simple_input = r"""
broadcaster -> a, b
%a -> ha
%b -> hb, c
%c -> ha
&ha -> ka, kc
&hb -> ka, kc
&ka -> kx
&kc -> kx
&kx -> rx
""".strip()


def test_p2_example2():
    assert d20.p2(simple_input) == 1
