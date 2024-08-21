from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cache
from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class NodeId(typing.Hashable, Protocol):
    pass


@dataclass(kw_only=True, slots=True)
class Digraph[Id: NodeId, N]:
    nodes: dict[Id, N]  # TODO: Consider replacing with a frozendict
    arcs: tuple[DigraphArc[Id], ...]

    def get_arcs_to(self, node_id: Id, /) -> list[DigraphArc[Id]]:
        return _get_arcs_to_node(node_id, self.arcs)

    def get_arcs_from(self, node_id: Id, /) -> list[DigraphArc[Id]]:
        return _get_arcs_from_node(node_id, self.arcs)


class DigraphArc[Id: NodeId](Protocol):
    @property
    def from_(self) -> Id: ...
    @property
    def to(self) -> Id: ...


@dataclass(frozen=True, slots=True)
class Arc[Id: NodeId]:
    from_: Id
    to: Id


class DigraphCreator[Id: NodeId, N]:
    def __init__(self) -> None:
        self._nodes: dict[Id, N] = {}
        self._arcs: list[DigraphArc[Id]] = []

    def add_node(self, node_id: Id, node: N, /) -> None:
        if node_id in self._nodes:
            raise ValueError(node_id)
        self._nodes[node_id] = node

    def add_arc(self, arc: DigraphArc[Id], /) -> None:
        if arc.from_ not in self._nodes:
            raise ValueError(arc.from_)
        if arc.to not in self._nodes:
            raise ValueError(arc.to)
        self._arcs.append(arc)

    def create(self) -> Digraph[Id, N]:
        return Digraph(nodes=self._nodes, arcs=tuple(self._arcs))


@cache
def _get_arcs_to_node[Id: NodeId](
    node_id: Id, arcs: Iterable[DigraphArc[Id]]
) -> list[DigraphArc[Id]]:
    return [arc for arc in arcs if arc.to == node_id]


@cache
def _get_arcs_from_node[Id: NodeId](
    node_id: Id, arcs: Iterable[DigraphArc[Id]]
) -> list[DigraphArc[Id]]:
    return [arc for arc in arcs if arc.from_ == node_id]
