import enum
from dataclasses import dataclass
from functools import total_ordering
from types import NotImplementedType
from typing import Iterable


def _parse_input(lines: Iterable[str]) -> Iterable[tuple[str, int]]:
    for line in lines:
        cards, bid = line.split()
        yield cards, int(bid)


@total_ordering
class HandType(enum.Enum):
    HighCard = 0
    OnePair = 1
    TwoPair = 2
    ThreeOfAKind = 3
    FullHouse = 4
    FourOfAKind = 5
    FiveOfAKind = 6

    def __lt__(self, other: "HandType") -> bool | NotImplementedType:
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


@dataclass
class Hand:
    card_values: list[int]
    hand_type: HandType
    bid: int

    def __lt__(self, other: "Hand") -> bool:
        if self.hand_type != other.hand_type:
            return self.hand_type < other.hand_type

        for i in range(5):
            if self.card_values[i] != other.card_values[i]:
                return self.card_values[i] < other.card_values[i]
        raise AssertionError()


def p1(input_str: str) -> int:
    d = _parse_input(input_str.splitlines())

    def classify_hand_type(cards: str) -> HandType:
        assert len(cards) == 5
        value_counts: dict[str, int] = {}
        for c in cards:
            value_counts[c] = value_counts.get(c, 0) + 1

        counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        if counts[0][1] == 5:
            return HandType.FiveOfAKind
        if counts[0][1] == 4:
            return HandType.FourOfAKind
        if counts[0][1] == 3:
            if counts[1][1] == 2:
                return HandType.FullHouse
            return HandType.ThreeOfAKind
        if counts[0][1] == 2:
            if counts[1][1] == 2:
                return HandType.TwoPair
            return HandType.OnePair
        return HandType.HighCard

    def card_value(value: str) -> int:
        assert len(value) == 1
        try:
            return int(value)
        except ValueError:
            pass

        return "TJQKA".index(value) + 10

    hands = [
        Hand(
            [card_value(c) for c in cards],
            classify_hand_type(cards),
            bid,
        )
        for cards, bid in d
    ]

    hands.sort()

    return sum((ind + 1) * hand.bid for ind, hand in enumerate(hands))


def p2(input_str: str) -> int:
    d = _parse_input(input_str.splitlines())

    def classify_hand_type(cards: str) -> HandType:
        assert len(cards) == 5
        value_counts: dict[str, int] = {}
        for c in cards:
            value_counts[c] = value_counts.get(c, 0) + 1

        jokers = value_counts.get("J", 0)
        value_counts.pop("J", None)

        if jokers == 5:
            return HandType.FiveOfAKind

        counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        if counts[0][1] + jokers == 5:
            return HandType.FiveOfAKind
        if counts[0][1] + jokers == 4:
            return HandType.FourOfAKind
        if counts[0][1] + jokers == 3:
            if counts[1][1] == 2:
                return HandType.FullHouse
            return HandType.ThreeOfAKind
        if counts[0][1] + jokers == 2:
            if counts[1][1] == 2:
                return HandType.TwoPair
            return HandType.OnePair
        return HandType.HighCard

    def card_value(value: str) -> int:
        assert len(value) == 1
        try:
            return int(value)
        except ValueError:
            pass
        if value == "J":
            return 1
        return "TQKA".index(value) + 10

    hands = [
        Hand(
            [card_value(c) for c in cards],
            classify_hand_type(cards),
            bid,
        )
        for cards, bid in d
    ]

    hands.sort()

    return sum((ind + 1) * hand.bid for ind, hand in enumerate(hands))
