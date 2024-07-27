import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from queue import Queue
from typing import Iterable, override

logger = logging.getLogger(__name__)

_Low = False
_High = True


# @dataclass(slots=True)
@dataclass
class _Module:
    name: str
    outputs: list[str] = field(default_factory=list)

    def add_input(self, name: str) -> None:
        # No-op by default
        pass

    def process_input(self, source_name: str, state: bool) -> bool | None:
        raise NotImplementedError()

    def get_diff(self, other: "_Module") -> dict[str, str]:
        assert self.__class__ == other.__class__
        assert self.name == other.name
        assert self.outputs == other.outputs
        return {}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.outputs})"


# @dataclass(slots=True)
@dataclass
class _FlipFlop(_Module):
    state: bool = _Low

    @override
    def process_input(self, source_name: str, state: bool) -> bool | None:
        if state is _High:
            return None

        self.state = not self.state
        return self.state

    @override
    def get_diff(self, other: _Module) -> dict[str, str]:
        diff = super().get_diff(other)
        assert isinstance(other, self.__class__)
        if self.state != other.state:
            diff["state"] = f"{self.state} -> {other.state}"
        return diff

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.outputs}, {self.state})"


# @dataclass(slots=True)
@dataclass
class _Conjuction(_Module):
    state: dict[str, bool] = field(default_factory=dict)

    @override
    def add_input(self, name: str) -> None:
        self.state[name] = _Low

    @override
    def process_input(self, source_name: str, state: bool) -> bool | None:
        assert source_name in self.state
        self.state[source_name] = state
        return not all(self.state.values())

    @override
    def get_diff(self, other: _Module) -> dict[str, str]:
        diff = super().get_diff(other)
        assert isinstance(other, self.__class__)
        assert self.state.keys() == other.state.keys()
        for name, state in self.state.items():
            if state != other.state[name]:
                diff[f"state[{name}]"] = f"{state} -> {other.state[name]}"
        return diff

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.outputs}, {self.state})"


# @dataclass(slots=True)
@dataclass
class _Broadcast(_Module):
    @override
    def process_input(self, source_name: str, state: bool) -> bool | None:
        return state


# @dataclass(slots=True)
@dataclass
class _Receiver(_Module):
    @override
    def process_input(self, source_name: str, state: bool) -> bool | None:
        return None


def _parse_module(line: str) -> _Module:
    name, outputs_str = map(str.strip, line.split("->"))
    outputs = list(map(str.strip, outputs_str.split(",")))
    if name == "broadcaster":
        return _Broadcast(name, outputs)

    assert name[0] in "%&"
    module_type = name[0]
    name = name[1:]

    if module_type == "%":
        return _FlipFlop(name, outputs)
    return _Conjuction(name, outputs)


def _parse_modules(lines: Iterable[str]) -> tuple[list[_Module], dict[str, _Module]]:
    modules = [_parse_module(line) for line in lines]
    modules_by_name = {module.name: module for module in modules}
    for module in modules:
        for output in module.outputs:
            output_module = modules_by_name.get(output)
            if output_module is None:
                receiver = _Receiver(output)
                modules.append(receiver)
                modules_by_name[output] = receiver
                output_module = receiver
            output_module.add_input(module.name)
    return modules, modules_by_name


@dataclass(slots=True, frozen=True)
class _SentPulse:
    source_name: str
    destination_name: str
    state: bool


def _log_state_diff(
    old_state: list[_Module], new_state: list[_Module], level: int = logging.DEBUG
) -> None:
    for old_module, new_module in zip(old_state, new_state):
        diff = old_module.get_diff(new_module)
        if diff:
            logger.log(level, "Module %s:", new_module)
            for key, value in diff.items():
                logger.log(level, "  %s: %s", key, value)


def _log_state(state: list[_Module]) -> None:
    for module in state:
        logger.debug("  %s", module)


def p1(input_str: str) -> int:
    modules, modules_by_name = _parse_modules(input_str.splitlines())
    modules.sort(key=lambda module: module.name)

    counts = {_Low: 0, _High: 0}
    initial_state = [deepcopy(module) for module in modules]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Initial state:")
        _log_state(modules)
        prev_state = deepcopy(modules)

    button_press_count = 0

    for button_press_count in range(1, 1001):
        sent_signals = Queue[_SentPulse]()
        sent_signals.put(_SentPulse("", "broadcaster", _Low))

        while sent_signals.empty() is False:
            signal = sent_signals.get_nowait()
            logger.debug("Processing signal: %s", signal)
            counts[signal.state] += 1
            destination = modules_by_name[signal.destination_name]
            output_signal_state = destination.process_input(
                signal.source_name, signal.state
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("State diff against previous state:")
                _log_state_diff(prev_state, modules)

            if output_signal_state is not None:
                for output in destination.outputs:
                    sent_signals.put_nowait(
                        _SentPulse(destination.name, output, output_signal_state)
                    )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Current state after %d button presses:", button_press_count)
            _log_state(modules)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "State diff against initial state after %d button presses:",
                button_press_count,
            )
            _log_state_diff(initial_state, modules, logging.INFO)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("State diff against previous state:")
            _log_state_diff(prev_state, modules)
            prev_state = deepcopy(modules)

        if modules == initial_state:
            logger.info(
                "Found initial state after %d button presses", button_press_count
            )
            break
        logger.info("Button press %d done", button_press_count)

    logger.info(f"Counts: {counts}")

    assert button_press_count > 0
    return math.prod(count * 1000 // button_press_count for count in counts.values())
