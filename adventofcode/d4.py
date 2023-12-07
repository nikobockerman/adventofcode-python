from dataclasses import dataclass
from typing import Iterable


@dataclass
class InputCards:
    card_id: int
    winning: list[int]
    own: list[int]


def _parse_input(lines: Iterable[str]) -> Iterable[InputCards]:
    for line in lines:
        card_id, numbers_str = line[4:].split(":")
        winning_str, own_str = numbers_str.strip().split("|")

        yield InputCards(
            int(card_id.strip()),
            [int(n) for n in winning_str.strip().split()],
            [int(n) for n in own_str.strip().split()],
        )


def p1(input_: str) -> int:
    d = _parse_input(input_.splitlines())

    result: int = 0
    for cards in d:
        matches = len(set(cards.winning) & set(cards.own))
        if matches > 0:
            result += 2 ** (matches - 1)
    return result


def p2(input_str: str) -> int:
    d = list(_parse_input(input_str.splitlines()))

    counts: dict[int, int] = {n + 1: 1 for n in range(len(d))}
    for cards in d:
        matches = len(set(cards.winning) & set(cards.own))
        card_count = counts[cards.card_id]
        for i in range(cards.card_id + 1, cards.card_id + 1 + matches):
            counts[i] += card_count

    return sum(v for v in counts.values())
