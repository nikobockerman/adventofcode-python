from dataclasses import dataclass
from typing import assert_type

from adventofcode.tooling.digraph import Arc, Digraph, DigraphCreator


def test_digraph_creator_simple() -> None:
    creator = DigraphCreator[int, int]()
    creator.add_node(1, 11)
    creator.add_node(2, 22)
    creator.add_arc(Arc(1, 2))
    digraph = creator.create()
    assert digraph.nodes == {1: 11, 2: 22}
    assert digraph.arcs == (Arc(1, 2),)
    assert_type(digraph.nodes, dict[int, int])


def test_digraph_creator_two_types() -> None:
    creator = DigraphCreator[int | str, int | str]()
    creator.add_node("a", "aa")
    creator.add_node(1, 11)
    creator.add_node("b", 22)
    creator.add_arc(Arc("a", 1))
    creator.add_arc(Arc("a", "b"))
    creator.add_arc(Arc(1, "b"))
    digraph = creator.create()
    assert digraph.nodes == {"a": "aa", 1: 11, "b": 22}
    assert digraph.arcs == (Arc("a", 1), Arc("a", "b"), Arc(1, "b"))
    assert_type(digraph.nodes, dict[int | str, int | str])


def test_digraph_creator_multiple_inherited_classes() -> None:
    @dataclass
    class Base:
        name: str

    class Child1(Base):
        pass

    class Child2(Base):
        pass

    creator = DigraphCreator[str, Child1 | Child2]()
    creator.add_node("a", Child1("a"))
    creator.add_node("b", Child2("b"))
    creator.add_arc(Arc("a", "b"))
    digraph = creator.create()
    assert digraph.nodes == {"a": Child1("a"), "b": Child2("b")}
    assert digraph.arcs == (Arc("a", "b"),)
    assert_type(digraph.nodes, dict[str, Child1 | Child2])


def test_digraph_get_arcs() -> None:
    digraph = Digraph[int, int](
        nodes={1: 11, 2: 22, 3: 33, 4: 44},
        arcs=tuple((Arc(1, 2), Arc(1, 3), Arc(2, 3), Arc(3, 1))),
    )
    assert digraph.get_arcs_from(1) == [Arc(1, 2), Arc(1, 3)]
    assert digraph.get_arcs_from(2) == [Arc(2, 3)]
    assert digraph.get_arcs_from(3) == [Arc(3, 1)]
    assert digraph.get_arcs_from(4) == []
    assert digraph.get_arcs_to(1) == [Arc(3, 1)]
    assert digraph.get_arcs_to(2) == [Arc(1, 2)]
    assert digraph.get_arcs_to(3) == [Arc(1, 3), Arc(2, 3)]
    assert digraph.get_arcs_to(4) == []


def test_digraph_weighted_arcs() -> None:
    @dataclass(frozen=True)
    class WeightedArc:
        from_: str
        to: str
        weight: int

    digraph_creator = DigraphCreator[str, int]()
    digraph_creator.add_node("a", 1)
    digraph_creator.add_node("b", 2)
    digraph_creator.add_node("c", 3)
    digraph_creator.add_arc(WeightedArc("a", "b", 3))
    digraph_creator.add_arc(WeightedArc("a", "c", 4))
    digraph_creator.add_arc(WeightedArc("b", "c", 5))
    digraph = digraph_creator.create()
    assert digraph.get_arcs_from("a") == [
        WeightedArc("a", "b", 3),
        WeightedArc("a", "c", 4),
    ]
    assert digraph.get_arcs_from("b") == [WeightedArc("b", "c", 5)]
    assert digraph.get_arcs_from("c") == []
    assert digraph.get_arcs_to("a") == []
    assert digraph.get_arcs_to("b") == [WeightedArc("a", "b", 3)]
    assert digraph.get_arcs_to("c") == [
        WeightedArc("a", "c", 4),
        WeightedArc("b", "c", 5),
    ]
