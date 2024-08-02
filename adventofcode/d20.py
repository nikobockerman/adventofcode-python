from __future__ import annotations

import itertools
import logging
import math
from abc import ABCMeta, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Iterator, Never, NewType, overload, override

_logger = logging.getLogger(__name__)


_PulseValue = NewType("_PulseValue", bool)
_PulseLow = _PulseValue(False)
_PulseHigh = _PulseValue(True)

_ModuleName = NewType("_ModuleName", str)


def to_str(value: _PulseValue) -> str:
    return "high" if value else "low"


@dataclass(frozen=True, kw_only=True, slots=True)
class _PulseNew:
    value: _PulseValue
    from_: _Module
    to: _Module


@dataclass(frozen=True, kw_only=True, slots=True)
class _Pulse:
    button_presses: int
    pulse_index: int
    value: _PulseValue
    from_: _Module
    to: _Module

    @staticmethod
    @overload
    def new(
        button_presses: int,
        pulse_index: int,
        value: _PulseValue,
        from_: _Module,
        to: _Module,
        /,
    ) -> _Pulse: ...

    @staticmethod
    @overload
    def new(
        button_presses: int, pulse_index: int, new_pulse: _PulseNew, /
    ) -> _Pulse: ...

    @staticmethod
    def new(
        button_presses: int,
        pulse_index: int,
        value_or_new_pulse: _PulseValue | _PulseNew,
        from_: _Module | None = None,
        to: _Module | None = None,
        /,
    ) -> _Pulse:
        if isinstance(value_or_new_pulse, _PulseNew):
            return _Pulse(
                button_presses=button_presses,
                pulse_index=pulse_index,
                value=value_or_new_pulse.value,
                from_=value_or_new_pulse.from_,
                to=value_or_new_pulse.to,
            )

        if from_ is None:
            raise ValueError("from_")
        if to is None:
            raise ValueError("to")
        return _Pulse(
            button_presses=button_presses,
            pulse_index=pulse_index,
            value=value_or_new_pulse,
            from_=from_,
            to=to,
        )


class _Module(metaclass=ABCMeta):
    def __init__(self, name: _ModuleName) -> None:
        self._name = name
        self._outputs: list[tuple[_Module, _PulseValue]] = []

    @property
    def name(self) -> _ModuleName:
        return self._name

    @property
    @abstractmethod
    def possible_output_values(self) -> tuple[_PulseValue, ...]: ...

    @property
    @abstractmethod
    def interesting_input_signal_values(self) -> tuple[_PulseValue, ...]: ...

    def add_receiving_module(self, output: _Module) -> None:
        interested_pulse_values = output.interesting_input_signal_values
        assert interested_pulse_values
        for value in interested_pulse_values:
            if value not in self.possible_output_values:
                continue

            self._outputs.append((output, value))

    @abstractmethod
    def process_pulse(self, pulse: _Pulse) -> Iterator[_PulseNew]: ...

    def _get_output_pulses(self, value: _PulseValue) -> Iterator[_PulseNew]:
        for output, interested_value in self._outputs:
            if interested_value == value:
                yield _PulseNew(value=value, from_=self, to=output)

    if __debug__:

        def _validate_incoming_pulse(self, pulse: _Pulse) -> None:
            assert pulse.to is self
            assert pulse.value in self.interesting_input_signal_values


class _Button(_Module):
    def __init__(self) -> None:
        super().__init__(name=_ModuleName(""))
        self._button_presses: int = 0

    @property
    def button_presses(self) -> int:
        return self._button_presses

    def process_button_press(self) -> Iterator[_PulseNew]:
        self._button_presses += 1
        yield from self._get_output_pulses(_PulseLow)

    @property
    @override
    def possible_output_values(self) -> tuple[_PulseValue, ...]:
        return (_PulseLow,)

    @property
    @override
    def interesting_input_signal_values(self) -> tuple[_PulseValue, ...]:
        return tuple()

    @override
    def process_pulse(self, pulse: _Pulse) -> Never:
        if __debug__:
            self._validate_incoming_pulse(pulse)
        raise AssertionError()


class _Broadcast(_Module):
    @property
    @override
    def possible_output_values(self) -> tuple[_PulseValue, ...]:
        return (_PulseLow, _PulseHigh)

    @property
    @override
    def interesting_input_signal_values(self) -> tuple[_PulseValue, ...]:
        return (_PulseLow, _PulseHigh)

    @override
    def process_pulse(self, pulse: _Pulse) -> Iterator[_PulseNew]:
        if __debug__:
            self._validate_incoming_pulse(pulse)
        yield from self._get_output_pulses(pulse.value)


class _Receiver(_Module):
    def __init__(self, name: _ModuleName) -> None:
        super().__init__(name=name)
        self._received_low = False

    @property
    def received_low(self) -> bool:
        return self._received_low

    @property
    @override
    def possible_output_values(self) -> tuple[_PulseValue, ...]:
        return tuple()

    @property
    @override
    def interesting_input_signal_values(self) -> tuple[_PulseValue, ...]:
        return (_PulseLow,)

    @override
    def add_receiving_module(self, output: _Module) -> Never:
        super().add_receiving_module(output)
        raise AssertionError()

    @override
    def process_pulse(self, pulse: _Pulse) -> Iterator[_PulseNew]:
        if __debug__:
            self._validate_incoming_pulse(pulse)
        _logger.info("Receiver received signal from %s: %s", pulse.from_, pulse.value)
        assert pulse.value is _PulseLow
        self._received_low = True
        yield from []


class _FlipFlop(_Module):
    def __init__(self, name: _ModuleName) -> None:
        super().__init__(name=name)
        self._state: _PulseValue = _PulseLow

    @property
    @override
    def possible_output_values(self) -> tuple[_PulseValue, ...]:
        return (_PulseLow, _PulseHigh)

    @property
    @override
    def interesting_input_signal_values(self) -> tuple[_PulseValue, ...]:
        return (_PulseLow,)

    @override
    def process_pulse(self, pulse: _Pulse) -> Iterator[_PulseNew]:
        if __debug__:
            self._validate_incoming_pulse(pulse)

        assert pulse.value is _PulseLow

        self._state = _PulseValue(not self._state)
        yield from self._get_output_pulses(self._state)


class _Conjunction(_Module):
    def __init__(self, name: _ModuleName) -> None:
        super().__init__(name=name)
        self._state: dict[_ModuleName, _PulseValue] = {}
        self._input_signals: dict[_ModuleName, list[tuple[_PulseValue, int]]] = {}

    @property
    @override
    def possible_output_values(self) -> tuple[_PulseValue, ...]:
        return (_PulseLow, _PulseHigh)

    @property
    @override
    def interesting_input_signal_values(self) -> tuple[_PulseValue, ...]:
        return (_PulseLow, _PulseHigh)

    @override
    def process_pulse(self, pulse: _Pulse) -> Iterator[_PulseNew]:
        if __debug__:
            self._validate_incoming_pulse(pulse)
        assert pulse.from_.name in self._state

        self._state[pulse.from_.name] = pulse.value

        output_pulse_value = (
            _PulseLow
            if all(value is _PulseHigh for value in self._state.values())
            else _PulseHigh
        )
        yield from self._get_output_pulses(output_pulse_value)

    def set_inputs(self, inputs: Iterable[_ModuleName]) -> None:
        self._state.update({name: _PulseLow for name in inputs})


@dataclass(frozen=True, slots=True, kw_only=True)
class Pattern:
    period: int


@dataclass(slots=True, kw_only=True)
class _PulseData:
    value: _PulseValue
    count: int
    first_button_press: int
    last_button_press: int

    @property
    def button_press_period(self) -> int:
        return self.last_button_press - self.first_button_press + 1

    def __repr__(self) -> str:
        return f"({self.value}, {self.count}, {self.first_button_press}, {self.last_button_press}, {self.button_press_period})"


class _GatewayConjuction(_Conjunction):
    def __init__(self, name: _ModuleName) -> None:
        super().__init__(name=name)
        self._input_pulses: dict[_ModuleName, list[_PulseData]] = {}
        self._patterns: dict[_ModuleName, Pattern] = {}

    @property
    def has_pattern_for_all(self) -> bool:
        return all(name in self._patterns for name in self._input_pulses)

    @property
    def patterns(self) -> dict[_ModuleName, Pattern]:
        return self._patterns

    @override
    def set_inputs(self, inputs: Iterable[_ModuleName]) -> None:
        inputs = list(inputs)
        super().set_inputs(inputs)
        self._input_pulses.update({name: [] for name in inputs})

    @override
    def process_pulse(self, pulse: _Pulse) -> Iterator[_PulseNew]:
        result_iter = super().process_pulse(pulse)

        input_pulses = self._input_pulses[pulse.from_.name]
        if not input_pulses or pulse.value != input_pulses[-1].value:
            input_pulses.append(
                _PulseData(
                    value=pulse.value,
                    count=0,
                    first_button_press=pulse.button_presses,
                    last_button_press=pulse.button_presses,
                )
            )
        input_pulses[-1].count += 1
        input_pulses[-1].last_button_press = pulse.button_presses

        if sum(data.count for data in input_pulses) % 200 == 0:
            _logger.info(
                "Conjuction(%s) input pulses for %s: %s",
                self.name,
                pulse.from_.name,
                input_pulses,
            )

        if pulse.from_.name not in self._patterns and len(input_pulses) == 7:
            _logger.info(
                "Conjuction(%s) input pulses for pattern: %s",
                self.name,
                input_pulses[:6],
            )
            first_low, first_high, second_low, second_high, third_low, third_high = (
                input_pulses[:6]
            )
            assert first_low.value is _PulseLow
            assert first_high.value is _PulseHigh
            assert second_low.value is _PulseLow
            assert second_high.value is _PulseHigh
            assert third_low.value is _PulseLow
            assert third_high.value is _PulseHigh
            assert first_low.count > 1
            assert first_high.count == 1
            assert second_low.count > 1
            assert second_high.count == 1
            assert third_low.count > 1
            assert third_high.count == 1

            assert first_low.first_button_press == 1
            assert first_low.last_button_press > 1
            assert first_low.button_press_period > 1

            assert first_high.first_button_press > first_low.last_button_press
            assert first_high.last_button_press == first_high.first_button_press
            assert first_high.button_press_period == 1

            assert second_low.first_button_press == first_high.last_button_press
            assert second_low.last_button_press > second_low.first_button_press
            assert second_low.button_press_period > 1

            assert second_high.first_button_press > second_low.last_button_press
            assert second_high.last_button_press == second_high.first_button_press
            assert second_high.button_press_period == 1

            assert third_low.first_button_press == second_high.last_button_press
            assert third_low.last_button_press > third_low.first_button_press
            assert third_low.button_press_period > 1

            assert third_high.first_button_press > third_low.last_button_press
            assert third_high.last_button_press == third_high.first_button_press
            assert third_high.button_press_period == 1

            assert first_low.count < second_low.count
            assert second_low.count == third_low.count

            assert first_low.button_press_period == second_low.button_press_period - 1
            assert second_low.button_press_period == third_low.button_press_period

            self._patterns[pulse.from_.name] = Pattern(
                period=second_low.button_press_period
            )
            _logger.info(
                "Conjuction(%s) pattern for %s: %s",
                self.name,
                pulse.from_.name,
                self._patterns[pulse.from_.name],
            )

        yield from result_iter

    # Pattern notes:
    # - Formula: start + n*constant
    # - start is always constant - 1 -> try without taking that into account
    # - Value changes to True and back to False during same button press: try first
    #   without taking this into consideration


type _AnyModule = (
    _Button | _Broadcast | _Receiver | _FlipFlop | _Conjunction | _GatewayConjuction
)


def _parse_module(line: str) -> tuple[_AnyModule, list[_ModuleName]]:
    name, outputs_str = map(str.strip, line.split("->"))
    name = _ModuleName(name)
    outputs = list(map(lambda x: _ModuleName(x.strip()), outputs_str.split(",")))
    if name == "broadcaster":
        return _Broadcast(name), outputs

    assert name[0] in "%&"
    module_type = name[0]
    name = _ModuleName(name[1:])

    if module_type == "%":
        return _FlipFlop(name), outputs
    return _Conjunction(name), outputs


def _parse_modules(
    lines: Iterable[str],
) -> tuple[
    _Receiver | None,
    _Button,
    _GatewayConjuction | None,
    dict[type[_AnyModule], list[_AnyModule]],
]:
    modules_with_output_names = [(_Button(), [_ModuleName("broadcaster")])] + [
        _parse_module(line) for line in lines
    ]

    all_names_with_outputs = {module.name for module, _ in modules_with_output_names}
    all_output_names = {
        output for _, outputs in modules_with_output_names for output in outputs
    }
    receiver_names = all_output_names - all_names_with_outputs
    assert (
        len(receiver_names) <= 1
    ), "Safety check: only inputs with 0-1 receivers are known"

    if receiver_names:
        receiver_name = receiver_names.pop()
        modules_with_output_names.append((_Receiver(receiver_name), []))
        gateways_to_receiver = [
            module
            for module, outputs in modules_with_output_names
            if receiver_name in outputs
        ]
        assert (
            len(gateways_to_receiver) <= 1
        ), "Safety check: only 0-1 gateways are known"
        if gateways_to_receiver:
            gateway_to_receiver = gateways_to_receiver[0]
            assert isinstance(
                gateway_to_receiver, _Conjunction
            ), "Safety check: only conjunctions are known"
            modules_with_output_names = [
                (
                    _GatewayConjuction(module.name)
                    if module is gateway_to_receiver
                    else module,
                    outputs,
                )
                for module, outputs in modules_with_output_names
            ]

    modules_by_name = {module.name: module for module, _ in modules_with_output_names}

    modules_with_output_modules = [
        (module, list(map(modules_by_name.__getitem__, outputs)))
        for module, outputs in modules_with_output_names
    ]

    for module, output_modules in modules_with_output_modules:
        for output_module in output_modules:
            module.add_receiving_module(output_module)

    modules_by_type: dict[type[_AnyModule], list[_AnyModule]] = {
        t: list(g)
        for t, g in itertools.groupby(
            sorted(
                list(
                    map(lambda x: x[0], modules_with_output_modules),
                ),
                key=lambda x: type(x).__name__,
            ),
            key=lambda x: type(x),
        )
    }

    for t, conjunctions in modules_by_type.items():
        if not issubclass(t, _Conjunction):
            continue
        for conjunction in conjunctions:
            assert isinstance(conjunction, _Conjunction)
            inputs = (
                module.name
                for module, outputs in modules_with_output_names
                if conjunction.name in outputs
            )
            conjunction.set_inputs(inputs)

    receiver = modules_by_type.get(_Receiver, [None])[0]
    if receiver is not None:
        assert isinstance(receiver, _Receiver)

    button = modules_by_type[_Button][0]
    assert isinstance(button, _Button)

    gateway = modules_by_type.get(_GatewayConjuction, [None])[0]
    if gateway is not None:
        assert isinstance(gateway, _GatewayConjuction)

    return receiver, button, gateway, modules_by_type


def p1(input_str: str) -> int:
    _, button, _, _ = _parse_modules(input_str.splitlines())

    counts = Counter[_PulseValue]()
    for _ in range(1000):
        queue = deque[_Pulse](button.process_button_press())
        while queue:
            pulse = queue.popleft()
            counts.update((pulse.value,))
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.debug("Pulse: %s (queue length: %d)", pulse, len(queue))
            queue.extend(pulse.to.process_pulse(pulse))

    _logger.info(f"Counts: {counts}")

    return math.prod(counts.values())


def _process_p2(input_str: str, output_module_name: _ModuleName) -> int:
    receiver, button, gateway, _ = _parse_modules(input_str.splitlines())
    assert receiver is not None
    assert receiver.name == output_module_name
    assert gateway is not None

    queue = deque[_Pulse]()
    pulse_index = -1

    def next_pulse_count() -> int:
        nonlocal pulse_index
        pulse_index += 1
        return pulse_index

    def new_pulse(new: _PulseNew) -> _Pulse:
        return _Pulse.new(button.button_presses, next_pulse_count(), new)

    def extend_pulses(pulses: Iterable[_PulseNew]) -> None:
        nonlocal queue
        queue.extend(map(new_pulse, pulses))

    while True:
        if not queue:
            extend_pulses(button.process_button_press())
            if button.button_presses % 100_000 == 0:
                _logger.info(f"Button presses: {button.button_presses:_}")

        while queue:
            pulse = queue.popleft()
            extend_pulses(pulse.to.process_pulse(pulse))

        if gateway.has_pattern_for_all:
            _logger.info(
                "All patterns found for gateway after %d button presses",
                button.button_presses,
            )
            break

    return math.lcm(*(pattern.period for pattern in gateway.patterns.values()))


def p2(input_str: str, output_module_name: _ModuleName | None = None) -> int:
    if output_module_name is None:
        output_module_name = _ModuleName("rx")

    return _process_p2(input_str, output_module_name)
