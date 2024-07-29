from __future__ import annotations

import enum
import itertools
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from queue import Queue
from typing import Iterable, Iterator, Literal, NewType, final, override

from adventofcode.tooling import digraph

_logger = logging.getLogger(__name__)


type _Signal = Literal["low", "high"]


_ModuleName = NewType("_ModuleName", str)
_PatternLength = NewType("_PatternLength", int)
_PatternIndex = NewType("_PatternIndex", int)


class _PatternAnalysisLogic(enum.Enum):
    AllNewButtonPressesPerModule = enum.auto()
    AllModulesInBatches = enum.auto()


class PostProcessLogic(enum.Enum):
    OnEveryButtonPress = enum.auto()
    InBatches = enum.auto()


class _PatternStartLogic(enum.Enum):
    AtZero = enum.auto()
    Anywhere = enum.auto()


class _PatternLengthLogic(enum.Enum):
    Exp2 = enum.auto()
    Any = enum.auto()


batch_length = 1_000
pattern_analysis_logic = _PatternAnalysisLogic.AllModulesInBatches
post_process_logic = PostProcessLogic.InBatches
pattern_start_logic = _PatternStartLogic.Anywhere
pattern_length_logic = _PatternLengthLogic.Any

pre_calculated_twos_exponents_for_lengths = [
    _PatternLength(i)
    for i in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
]


@dataclass
class _Record:
    source: _ModuleName
    source_signal: _Signal
    resulting_signal: _Signal | None


@dataclass
class _Pattern:
    start_button_press_count: int
    records: list[list[_Record]]


@dataclass
class _PatternDetector:
    module_name: _ModuleName
    required_inputs: frozenset[_ModuleName]
    required_signals: frozenset[_Signal]
    required_signals_count: int
    signals: list[list[_Record]] = field(default_factory=list, init=False)
    detected_pattern: _Pattern | None = field(default=None, init=False)
    detected_final_result: bool = field(default=False, init=False)
    _checked_starts: dict[_PatternLength, _PatternIndex] = field(
        default_factory=dict, init=False
    )
    _checked_length: _PatternLength = field(default=_PatternLength(0), init=False)

    def record_processed_input(
        self,
        button_press_count: int,
        source: _ModuleName,
        source_signal: _Signal,
        resulting_signal: _Signal | None,
    ) -> None:
        if self.detected_pattern is not None and self.detected_final_result is True:
            return
        assert source in self.required_inputs
        assert len(self.signals) <= button_press_count
        appended = False
        while len(self.signals) < button_press_count:
            self.signals.append([])
            appended = True
        self.signals[-1].append(_Record(source, source_signal, resulting_signal))
        if (
            pattern_analysis_logic is _PatternAnalysisLogic.AllNewButtonPressesPerModule
            and self.detected_pattern is None
            and appended is True
        ):
            self.detected_pattern = self._detect_pattern()

    def check_final_pattern(self) -> bool:
        assert self.detected_pattern is not None
        assert self.detected_final_result is False
        pattern = self._detect_pattern(check_all=True)
        if self.detected_pattern != pattern:
            self.detected_pattern = pattern
        return pattern is not None

    def process_pattern_detection(self) -> None:
        if self.detected_pattern is not None:
            return
        self.detected_pattern = self._detect_pattern()

    def _detect_pattern(self, *, check_all: bool = False) -> _Pattern | None:
        start_time = time.perf_counter()
        try:
            signals = self.signals[:-1]
            # Check whether all required inputs are present
            # all_signal_sources = frozenset(
            #    record.source for record in itertools.chain(*signals)
            # )
            # if self.required_inputs != all_signal_sources:
            #    _logger.debug(
            #        "%s, Skip detection: not all inputs present in signals",
            #        self.module_name,
            #    )
            #    return None

            required_inputs = Counter(
                {
                    input_: self.required_signals_count
                    for input_ in set(
                        itertools.product(self.required_inputs, self.required_signals)
                    )
                }
            )

            # Check that all required sources are present in signals
            contained_inputs = Counter[tuple[_ModuleName, _Signal]]()
            for record in itertools.chain(*signals):
                contained_inputs.update(((record.source, record.source_signal),))
                if required_inputs == contained_inputs & required_inputs:
                    break
            else:
                _logger.debug(
                    "%s, Skip detection: not all required input signals present",
                    self.module_name,
                )
                return None

            # Detect if signals contains two consecutive sublist with same contents
            max_sublist_length = len(signals) // 2

            if check_all:
                lengths = range(2, max_sublist_length + 1)
            else:
                if pattern_length_logic is _PatternLengthLogic.Exp2:

                    def exp_lengths() -> Iterator[int]:
                        assert len(pre_calculated_twos_exponents_for_lengths) == 14
                        yield from pre_calculated_twos_exponents_for_lengths
                        for exp in itertools.count(
                            len(pre_calculated_twos_exponents_for_lengths) + 1
                        ):
                            yield 2**exp

                    it = exp_lengths()

                    if pattern_start_logic is _PatternStartLogic.AtZero:
                        it = itertools.dropwhile(
                            lambda length: length <= self._checked_length, it
                        )

                    lengths = itertools.takewhile(
                        lambda length: length <= max_sublist_length, it
                    )

                elif pattern_length_logic is _PatternLengthLogic.Any:
                    if pattern_start_logic is _PatternStartLogic.AtZero:
                        min_sublist_length = max(2, self._checked_length + 1)
                    elif pattern_start_logic is _PatternStartLogic.Anywhere:
                        min_sublist_length = 2
                    else:
                        raise AssertionError()
                    lengths = range(min_sublist_length, max_sublist_length + 1)
                else:
                    raise AssertionError()

            if pattern_start_logic is _PatternStartLogic.AtZero and not check_all:
                lengths = list(lengths)
                if lengths:
                    max_length = lengths[-1]
                    assert max_length == max(lengths)
                    assert max_length >= self._checked_length
                    self._checked_length = _PatternLength(max_length)

            if __debug__:
                lengths = list(lengths)
                if lengths:
                    min_length = lengths[0]
                    assert min_length == min(lengths)
                    assert min_length >= 2

            lengths = map(_PatternLength, lengths)
            # lengths = list(lengths)
            # _logger.debug(
            #    "%s: Detect lenghts: %d, %s", self.module_name, len(lengths), lengths
            # )

            for length in lengths:
                if check_all:
                    min_sublist1_start = _PatternIndex(0)
                    max_sublist1_start = _PatternIndex(len(signals) - 2 * length)
                elif pattern_start_logic is _PatternStartLogic.AtZero:
                    min_sublist1_start = _PatternIndex(0)
                    max_sublist1_start = _PatternIndex(0)
                elif pattern_start_logic is _PatternStartLogic.Anywhere:
                    min_sublist1_start = _PatternIndex(
                        self._checked_starts.get(length, -1) + 1
                    )
                    max_sublist1_start = _PatternIndex(len(signals) - 2 * length)
                    if not check_all:
                        self._checked_starts[length] = max_sublist1_start
                else:
                    raise AssertionError()

                # Check that all required sources are present in signals[min, max]
                if (
                    min_sublist1_start > 0
                ):  # If 0, then the check outside this loop is enough
                    contained_inputs = Counter[tuple[_ModuleName, _Signal]]()
                    for record in itertools.chain(
                        *signals[min_sublist1_start : max_sublist1_start + length]
                    ):
                        contained_inputs.update(
                            ((record.source, record.source_signal),)
                        )
                        if required_inputs == contained_inputs & required_inputs:
                            break
                    else:
                        # _logger.debug(
                        #    "%s, %d, Skip length detection: not all required input signals",
                        #    self.module_name,
                        #    length,
                        # )
                        continue

                starts = map(
                    _PatternIndex,
                    range(min_sublist1_start, max_sublist1_start + 1),
                )
                # _logger.debug("%s, %d: Detect starts: %s", self.module_name, length, starts)

                for start in starts:
                    # _logger.debug(
                    #    "Detecting pattern: module:%s, start:%d, length:%d, signals len:%d",
                    #    self.module_name,
                    #    start,
                    #    length,
                    #    len(signals),
                    # )
                    start1 = start
                    end1 = start1 + length
                    sublist1 = signals[start1:end1]

                    # Check that sublist1 isn't completely empty of signals
                    # if all(not signals for signals in sublist1):
                    #    continue

                    # Check that all required sources are present in sublist1
                    contained_inputs = Counter[tuple[_ModuleName, _Signal]]()
                    for record in itertools.chain(*sublist1):
                        contained_inputs.update(
                            ((record.source, record.source_signal),)
                        )
                        if required_inputs == contained_inputs & required_inputs:
                            break
                    else:
                        continue

                    remaining_sublists_count = (len(signals) - end1) // length
                    assert remaining_sublists_count >= 1
                    if not check_all:
                        # During normal execution, only check first following sublist
                        remaining_sublists_count = 1
                    all_sublists_match = True
                    for start_other in map(
                        _PatternIndex,
                        range(end1, end1 + remaining_sublists_count * length, length),
                    ):
                        end_other = _PatternIndex(start_other + length)
                        assert end_other <= len(signals)

                        sublist_other = signals[start_other:end_other]

                        # Check that sublist1 and sublist_other have same contents
                        if sublist1 != sublist_other:
                            all_sublists_match = False
                            break

                    if all_sublists_match:
                        _logger.info(
                            "Found pattern for %s: start=%d, len=%d: %s",
                            self.module_name,
                            start1,
                            len(sublist1),
                            sublist1,
                        )
                        if start1 != 0:
                            raise AssertionError("Proof that start isn't always 0")  # noqa: TRY003
                        if (
                            len(sublist1)
                            > pre_calculated_twos_exponents_for_lengths[-1]
                        ):
                            raise AssertionError(  # noqa: TRY003
                                "Need to pre calculate more 2**x lenghts"
                            )
                        if (
                            len(sublist1)
                            not in pre_calculated_twos_exponents_for_lengths
                        ):
                            raise AssertionError(  # noqa: TRY003
                                "Proof that pattern length isn't always 2**x"
                            )
                        return _Pattern(start1, sublist1)

            _logger.debug("%s, No pattern found", self.module_name)
            return None
        finally:
            end = time.perf_counter()
            _logger.debug("%s, Took %f seconds", self.module_name, end - start_time)


# class _Module(digraph.DigraphNode):
@dataclass(eq=False)
class _Module:
    name: _ModuleName
    pattern_detector: _PatternDetector | None = None
    required_input_signals: frozenset[_Signal] = field(
        default=frozenset(("low",)), init=False
    )
    required_input_signals_count: int = field(default=1, init=False)

    def set_inputs(self, inputs: Iterable[_ModuleName]) -> None:
        self.pattern_detector = _PatternDetector(
            self.name,
            frozenset(inputs),
            self.required_input_signals,
            self.required_input_signals_count,
        )

    @final
    def process_input(
        self, button_press_count: int, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        result = self._process_input_internal(source_name, state)
        if self.pattern_detector is not None:
            self.pattern_detector.record_processed_input(
                button_press_count, source_name, state, result
            )
        return result

    def _process_input_internal(
        self, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        raise NotImplementedError()

    def get_diff(self, other: _Module) -> dict[str, str]:
        assert self.__class__ == other.__class__
        assert self.name == other.name
        return {}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


@dataclass()
class _FlipFlop(_Module):
    state: _Signal = field(default="low", init=False)
    required_input_signals_count: int = field(default=2, init=False)

    @override
    def _process_input_internal(
        self, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        if state == "high":
            return None

        if self.state == "low":
            self.state = "high"
        elif self.state == "high":
            self.state = "low"
        else:
            raise AssertionError(f"Unexpected state: {self.state}")  # noqa: TRY003
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
        return f"{self.__class__.__name__}({self.name}, {self.state})"


@dataclass()
class _Conjuction(_Module):
    state: dict[_ModuleName, _Signal] = field(default_factory=dict, init=False)
    required_input_signals: frozenset[_Signal] = field(
        default=frozenset(("low", "high")), init=False
    )

    @override
    def set_inputs(self, inputs: Iterable[_ModuleName]) -> None:
        inputs = list(inputs)
        super().set_inputs(inputs)
        self.state.update({name: "low" for name in inputs})

    @override
    def _process_input_internal(
        self, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        assert source_name in self.state
        self.state[source_name] = state
        return (
            "low" if all(value == "high" for value in self.state.values()) else "high"
        )

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
        return f"{self.__class__.__name__}({self.name}, {self.state})"


@dataclass()
class _Broadcast(_Module):
    @override
    def set_inputs(self, inputs: Iterable[_ModuleName]) -> None:
        inputs = list(inputs)
        assert len(inputs) == 0
        super().set_inputs((_ModuleName(""),))

    @override
    def _process_input_internal(
        self, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        return state


@dataclass()
class _Receiver(_Module):
    @override
    def _process_input_internal(
        self, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        if state == "low":
            _logger.warning("Receiver received signal from %s: %s", source_name, state)
        return None


type AnyModule = _FlipFlop | _Conjuction | _Broadcast | _Receiver


def _parse_module(line: str) -> tuple[AnyModule, set[_ModuleName]]:
    name, outputs_str = map(str.strip, line.split("->"))
    name = _ModuleName(name)
    outputs = set(map(lambda x: _ModuleName(x.strip()), outputs_str.split(",")))
    if name == "broadcaster":
        return _Broadcast(name), outputs

    assert name[0] in "%&"
    module_type = name[0]
    name = _ModuleName(name[1:])

    if module_type == "%":
        return _FlipFlop(name), outputs
    return _Conjuction(name), outputs


def _parse_modules(lines: Iterable[str]) -> digraph.Digraph[_ModuleName, AnyModule]:
    digraph_creator = digraph.DigraphCreator[_ModuleName, AnyModule]()

    modules_and_outputs = [_parse_module(line) for line in lines]
    assert all(module.name != "rx" for module, _ in modules_and_outputs)

    all_modules_that_have_outputs = {module.name for module, _ in modules_and_outputs}
    all_output_modules = {
        output for _, outputs in modules_and_outputs for output in outputs
    }
    receiver_modules = all_output_modules - all_modules_that_have_outputs
    assert (
        len(receiver_modules) <= 1
    ), "Safety check: only inputs with 0-1 receivers are known"
    for receiver in receiver_modules:
        digraph_creator.add_node(receiver, _Receiver(receiver))

    for module, _ in modules_and_outputs:
        digraph_creator.add_node(module.name, module)

    for module, outputs in modules_and_outputs:
        for output in outputs:
            digraph_creator.add_arc(digraph.Arc(module.name, output))

    graph = digraph_creator.create()

    for name, module in graph.nodes.items():
        module.set_inputs(map(lambda arc: arc.from_, graph.get_arcs_to(name)))

    return graph


@dataclass(slots=True, frozen=True)
class _SentPulse:
    source: _ModuleName
    destination: _ModuleName
    state: _Signal


def _log_state(state: Iterable[_Module]) -> None:
    for module in state:
        _logger.debug("  %s", module)


@dataclass
class _ButtonPressProcessInput:
    graph: digraph.Digraph[_ModuleName, AnyModule]
    button_press_count: int


@dataclass
class _ButtonPressProcessResult:
    counts: Counter[_Signal]
    signals: list[_SentPulse]


class _RecordQueue(Queue[_SentPulse]):
    def __init__(self) -> None:
        super().__init__()
        self.record: list[_SentPulse] = []

    @override
    def put(
        self, item: _SentPulse, block: bool = True, timeout: float | None = None
    ) -> None:
        self.record.append(item)
        super().put(item, block, timeout)


def _process_button_press(
    input_: _ButtonPressProcessInput,
) -> _ButtonPressProcessResult:
    signal_queue = _RecordQueue()
    signal_queue.put(_SentPulse(_ModuleName(""), _ModuleName("broadcaster"), "low"))

    counts = Counter[_Signal]()

    while signal_queue.empty() is False:
        signal = signal_queue.get_nowait()
        # _logger.debug("Processing signal: %s", signal)
        counts.update((signal.state,))
        destination = input_.graph.nodes[signal.destination]
        output_signal_state = destination.process_input(
            input_.button_press_count, signal.source, signal.state
        )

        if output_signal_state is not None:
            signal_source = destination
            for output in map(
                lambda arc: arc.to, input_.graph.get_arcs_from(destination.name)
            ):
                signal_queue.put_nowait(
                    _SentPulse(signal_source.name, output, output_signal_state)
                )

    return _ButtonPressProcessResult(counts, signal_queue.record)


def p1(input_str: str) -> int:
    graph = _parse_modules(input_str.splitlines())
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug("Initial state:")
        _log_state(graph.nodes.values())

    counts = Counter[_Signal]()
    for button_press_count in range(1, 1001):
        result = _process_button_press(
            _ButtonPressProcessInput(graph, button_press_count)
        )
        # Add counts from result to local dict
        counts.update(result.counts)

    _logger.info(f"Counts: {counts}")

    return math.prod(counts.values())


def _resolve_cycles(
    graph: digraph.Digraph[_ModuleName, AnyModule],
    visited_nodes: tuple[_ModuleName, ...],
    name: _ModuleName,
    to_reach: _ModuleName,
) -> Iterator[tuple[_ModuleName, ...]]:
    visited_nodes = (*visited_nodes, name)
    for to in map(lambda arc: arc.to, graph.get_arcs_from(name)):
        if to == to_reach:
            yield visited_nodes
        if to in visited_nodes:
            continue
        yield from _resolve_cycles(graph, visited_nodes, to, to_reach)


def _get_cycles(
    graph: digraph.Digraph[_ModuleName, AnyModule],
) -> list[frozenset[_ModuleName]]:
    cycles: list[frozenset[_ModuleName]] = []
    for name in graph.nodes:
        for cycle in map(
            lambda cycle: frozenset(cycle), _resolve_cycles(graph, (), name, name)
        ):
            if cycle in cycles:
                continue
            cycles.append(cycle)
    return cycles


def p2(input_str: str, output_module_name: _ModuleName | None = None) -> int:
    if output_module_name is None:
        output_module_name = _ModuleName("rx")

    graph = _parse_modules(input_str.splitlines())

    cycles = _get_cycles(graph)

    for cycle in cycles:
        _logger.warning("Cycle detected: %s", cycle)

    nodes_in_cycles = set(itertools.chain.from_iterable(cycles))
    _logger.warning("Count of nodes in total:  %d", len(graph.nodes))
    _logger.warning("Count of nodes in cycles: %d", len(nodes_in_cycles))
    nodes_not_in_cycles = set(graph.nodes) - nodes_in_cycles
    _logger.warning("Nodes not in cycles: %s", nodes_not_in_cycles)

    map_of_nodes_to_nodes_in_same_cycle = dict[_ModuleName, set[_ModuleName]]()
    for name in nodes_in_cycles:
        in_cycles = {node for cycle in cycles if name in cycle for node in cycle}
        map_of_nodes_to_nodes_in_same_cycle[name] = in_cycles

    button_press_count: int = 0
    nodes_with_finalized_pattern = set[_ModuleName]()
    nodes_with_considered_pattern = set[_ModuleName]()

    def detect_individual_patterns(nodes: set[_ModuleName]) -> set[_ModuleName]:
        nodes_detected_with_considered_pattern = set[_ModuleName]()
        nodes_to_process = sorted(
            nodes - nodes_with_finalized_pattern - nodes_with_considered_pattern
        )
        for node in nodes_to_process:
            module = graph.nodes[node]
            detector = module.pattern_detector
            assert detector is not None
            if pattern_analysis_logic is _PatternAnalysisLogic.AllModulesInBatches:
                detector.process_pattern_detection()
            if detector.detected_pattern is None:
                continue

            if node in map_of_nodes_to_nodes_in_same_cycle:
                nodes_detected_with_considered_pattern.add(node)
                continue

            assert node in nodes_not_in_cycles
            assert detector.detected_final_result is False
            from_nodes = set(map(lambda arc: arc.from_, graph.get_arcs_to(node)))
            if not from_nodes <= nodes_with_finalized_pattern:
                continue

            is_final = detector.check_final_pattern()
            assert is_final is True
            detector.detected_final_result = True
            assert node not in nodes_with_considered_pattern
            nodes_with_finalized_pattern.add(node)
        return nodes_detected_with_considered_pattern

    def get_loggable_node(node: _ModuleName) -> tuple[str, _ModuleName]:
        module = graph.nodes[node]
        return (type(module).__name__, node)

    def get_loggable_nodes(
        nodes: Iterable[_ModuleName],
    ) -> dict[str, list[_ModuleName]]:
        return {
            t: list(map(lambda x: x[1], g))
            for t, g in itertools.groupby(
                sorted(list(map(get_loggable_node, nodes))), key=lambda x: x[0]
            )
        }

    for button_press_count in itertools.count(1):
        if (
            _logger.isEnabledFor(logging.INFO)
            and button_press_count % batch_length == 0
        ):
            _logger.info("Button pressed. Count: %d", button_press_count)
        result = _process_button_press(
            _ButtonPressProcessInput(graph, button_press_count)
        )

        if post_process_logic is PostProcessLogic.OnEveryButtonPress:
            destinations = {signal.destination for signal in result.signals}
            nodes_detected_with_considered_pattern = detect_individual_patterns(
                destinations
            )
        elif post_process_logic is PostProcessLogic.InBatches:
            if button_press_count % batch_length != 0:
                continue

            nodes = set(graph.nodes)
            nodes_detected_with_considered_pattern = detect_individual_patterns(nodes)
        else:
            raise AssertionError()

        nodes_with_considered_pattern |= nodes_detected_with_considered_pattern
        nodes_with_some_pattern = (
            nodes_with_finalized_pattern | nodes_with_considered_pattern
        )
        if button_press_count % batch_length == 0:
            # Log in batches regardless of post_process_logic
            _logger.info(
                "Modules with finalized pattern: %d, %s",
                len(nodes_with_finalized_pattern),
                get_loggable_nodes(nodes_with_finalized_pattern),
            )
            _logger.info(
                "Modules with considered pattern: %d, %s",
                len(nodes_with_considered_pattern),
                get_loggable_nodes(nodes_with_considered_pattern),
            )
            nodes_without_any_pattern = set(graph.nodes) - nodes_with_some_pattern
            _logger.info(
                "Modules without any pattern: %d, %s",
                len(nodes_without_any_pattern),
                get_loggable_nodes(nodes_without_any_pattern),
            )

        nodes_to_process = set(nodes_detected_with_considered_pattern)
        while nodes_to_process:
            node = nodes_to_process.pop()
            detector = graph.nodes[node].pattern_detector
            assert detector is not None

            direct_from_nodes = map_of_nodes_to_nodes_in_same_cycle[node]
            if not direct_from_nodes <= nodes_with_some_pattern:
                nodes_to_process -= direct_from_nodes
                continue

            def collect_modules_in_same_cycles(node: _ModuleName) -> set[_ModuleName]:
                nodes = set[_ModuleName]()
                nodes_to_collect = {node}
                while nodes_to_collect:
                    node = nodes_to_collect.pop()
                    nodes.add(node)
                    nodes_to_collect |= (
                        map_of_nodes_to_nodes_in_same_cycle[node] - nodes
                    )
                return nodes

            nodes_in_shared_cycles = collect_modules_in_same_cycles(node)
            nodes_to_process -= nodes_in_shared_cycles
            if not nodes_in_shared_cycles <= nodes_with_some_pattern:
                continue

            def is_ready_for_final_pattern(node: _ModuleName) -> bool:
                detector = graph.nodes[node].pattern_detector
                assert detector is not None
                ready = detector.check_final_pattern()
                assert ready is True
                return ready

            if not all(map(is_ready_for_final_pattern, nodes_in_shared_cycles)):
                continue

            for node in nodes_in_shared_cycles:
                detector = graph.nodes[node].pattern_detector
                assert detector is not None
                detector.detected_final_result = True
                assert node not in nodes_with_finalized_pattern
                assert node in nodes_with_considered_pattern
                nodes_with_finalized_pattern.add(node)
                nodes_with_considered_pattern.remove(node)

        if any(
            signal.destination == output_module_name and signal.state == "low"
            for signal in result.signals
        ):
            _logger.info(
                "Low signal sent to '%s' after %d button presses",
                output_module_name,
                button_press_count,
            )
            break

    return button_press_count
