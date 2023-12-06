from dataclasses import dataclass, field
import logging
from typing import Iterable


@dataclass
class InputNumber:
    value: str
    begin_index: int


@dataclass
class InputSymbol:
    symbol: str
    index: int


@dataclass
class InputRow:
    numbers: list[InputNumber] = field(default_factory=list, init=False)
    symbols: list[InputSymbol] = field(default_factory=list, init=False)


def _parse_input(lines: Iterable[str]) -> list[InputRow]:
    rows: list[InputRow] = []
    for row_ind, line in enumerate(lines):
        logging.debug(f"{row_ind}: {line=}")
        row = InputRow()

        number: str | None = None
        for ind, symbol in enumerate(line):
            if symbol.isdigit():
                if number is None:
                    number = symbol
                else:
                    number += symbol
                continue

            elif number is not None:
                row.numbers.append(InputNumber(number, ind - len(number)))
                number = None

            if symbol == ".":
                continue

            row.symbols.append(InputSymbol(symbol, ind))

        if number is not None:
            row.numbers.append(InputNumber(number, len(line) - 1 - len(number)))
            number = None

        logging.debug(f"{row_ind}: {row=}")
        rows.append(row)
    return rows


def p1(input: str) -> int:
    d = _parse_input(input.splitlines())

    symbol_indexes = [[symbol.index for symbol in row.symbols] for row in d]

    @dataclass
    class Number:
        number: int
        adjacent_ind_range_begin: int
        adjacent_ind_range_end: int
        adjacent_row_range_begin: int
        adjacent_row_range_end: int

    def is_adjacent(number: Number) -> bool:
        for row_ind in range(
            number.adjacent_row_range_begin, number.adjacent_row_range_end
        ):
            for ind in range(
                number.adjacent_ind_range_begin, number.adjacent_ind_range_end
            ):
                if ind in symbol_indexes[row_ind]:
                    return True
        return False

    sum = 0
    for row_ind, row in enumerate(d):
        for input_number in row.numbers:
            number = Number(
                int(input_number.value),
                input_number.begin_index - 1,
                input_number.begin_index + len(input_number.value) + 1,
                max(0, row_ind - 1),
                min(len(d), row_ind + 2),
            )
            if is_adjacent(number):
                logging.debug(f"{input_number=}. Adjacent")
                sum += number.number
            else:
                logging.debug(f"{input_number=}. NOT adjacent")

    return sum


def p2(input: str) -> int:
    d = _parse_input(input.splitlines())

    @dataclass
    class Number:
        number: int
        adjacent_ind_range_begin: int
        adjacent_ind_range_end: int
        adjacent_row_range_begin: int
        adjacent_row_range_end: int

    @dataclass
    class GearSymbol:
        row_ind: int
        ind: int

    gear_symbols = [
        GearSymbol(row_ind, symbol.index)
        for row_ind, row in enumerate(d)
        for symbol in row.symbols
        if symbol.symbol == "*"
    ]

    numbers: list[Number] = [
        Number(
            int(input_number.value),
            input_number.begin_index - 1,
            input_number.begin_index + len(input_number.value) + 1,
            max(0, row_ind - 1),
            min(len(d), row_ind + 2),
        )
        for row_ind, row in enumerate(d)
        for input_number in row.numbers
    ]

    def is_adjacent(number: Number, gear_symbol: GearSymbol) -> bool:
        if gear_symbol.row_ind not in range(
            number.adjacent_row_range_begin, number.adjacent_row_range_end
        ):
            return False
        if gear_symbol.ind not in range(
            number.adjacent_ind_range_begin, number.adjacent_ind_range_end
        ):
            return False
        return True

    sum = 0
    for gear_symbol in gear_symbols:
        adjacent_numbers: list[Number] = []
        for number in numbers:
            if is_adjacent(number, gear_symbol):
                adjacent_numbers.append(number)
                if len(adjacent_numbers) > 2:
                    break
        if len(adjacent_numbers) == 2:
            sum += adjacent_numbers[0].number * adjacent_numbers[1].number
    return sum
