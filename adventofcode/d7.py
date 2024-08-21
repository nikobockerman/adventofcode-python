import enum
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import total_ordering
from types import NotImplementedType


def _parse_input(lines: Iterable[str]) -> Iterable[tuple[str, int]]:
    for line in lines:
        cards, bid = line.split()
        yield cards, int(bid)


@total_ordering
class _HandType(enum.Enum):
    HighCard = 0
    OnePair = 1
    TwoPair = 2
    ThreeOfAKind = 3
    FullHouse = 4
    FourOfAKind = 5
    FiveOfAKind = 6

    def __lt__(self, other: "_HandType") -> bool | NotImplementedType:
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


@dataclass
class _Hand:
    card_values: list[int]
    hand_type: _HandType
    bid: int

    def __lt__(self, other: "_Hand") -> bool:
        if self.hand_type != other.hand_type:
            return self.hand_type < other.hand_type

        for i in range(5):
            if self.card_values[i] != other.card_values[i]:
                return self.card_values[i] < other.card_values[i]
        raise AssertionError()


def p1(input_str: str) -> int:
    d = _parse_input(input_str.splitlines())

    def classify_hand_type(cards: str) -> _HandType:
        assert len(cards) == 5
        value_counts = Counter[str](cards)
        counts = [x[1] for x in value_counts.most_common(2)]

        if counts[0] == 5:
            return _HandType.FiveOfAKind
        if counts[0] == 4:
            return _HandType.FourOfAKind
        if counts[0] == 3:
            if counts[1] == 2:
                return _HandType.FullHouse
            return _HandType.ThreeOfAKind
        if counts[0] == 2:
            if counts[1] == 2:
                return _HandType.TwoPair
            return _HandType.OnePair
        return _HandType.HighCard

    def card_value(value: str) -> int:
        assert len(value) == 1
        try:
            return int(value)
        except ValueError:
            pass

        return "TJQKA".index(value) + 10

    hands = [
        _Hand(
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

    def classify_hand_type(cards: str) -> _HandType:
        assert len(cards) == 5
        value_counts = Counter[str](cards)

        jokers = value_counts.get("J", 0)
        value_counts.pop("J", None)

        if jokers == 5:
            return _HandType.FiveOfAKind

        counts = [x[1] for x in value_counts.most_common(2)]
        if counts[0] + jokers == 5:
            return _HandType.FiveOfAKind
        if counts[0] + jokers == 4:
            return _HandType.FourOfAKind
        if counts[0] + jokers == 3:
            if counts[1] == 2:
                return _HandType.FullHouse
            return _HandType.ThreeOfAKind
        if counts[0] + jokers == 2:
            if counts[1] == 2:
                return _HandType.TwoPair
            return _HandType.OnePair
        return _HandType.HighCard

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
        _Hand(
            [card_value(c) for c in cards],
            classify_hand_type(cards),
            bid,
        )
        for cards, bid in d
    ]

    hands.sort()

    return sum((ind + 1) * hand.bid for ind, hand in enumerate(hands))
