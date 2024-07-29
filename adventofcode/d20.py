from __future__ import annotations

import enum
import itertools
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from queue import Queue
from typing import Iterable, Iterator, Literal, NewType, assert_never, final, override

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


pre_calculated_twos_exponents_for_lengths = [
    _PatternLength(i)
    for i in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
]


@dataclass(frozen=True, slots=True)
class _Record:
    source: _ModuleName
    source_signal: _Signal
    resulting_signal: _Signal | None


@dataclass(frozen=True, slots=True)
class _Pattern:
    start_button_press_count: int
    records: list[list[_Record]]


type _CheckType = Literal["full", "minimal"]


@dataclass(kw_only=True, frozen=True, slots=True)
class _DetectPatternInputFull:
    module_name: _ModuleName
    required_input_signals: Counter[tuple[_ModuleName, _Signal]]
    signals: list[list[_Record]]


@dataclass(kw_only=True, frozen=True, slots=True)
class _DetectPatternInputMinimal(_DetectPatternInputFull):
    previously_processed_max_length: _PatternLength | None
    previously_processed_max_starts: dict[_PatternLength, _PatternIndex]


type _DetectPatternInput = _DetectPatternInputFull | _DetectPatternInputMinimal

_Duration = NewType("_Duration", float)
type _SkipAllReason = Literal["missing_required_inputs"]
type _PatternLengthSkipReason = Literal["not_found", "missing_required_inputs"]
type _PatternLengthAndStartNotFoundReason = Literal[
    "missing_required_inputs", "following_sublist_mismatch"
]


@dataclass(kw_only=True, frozen=True, slots=True)
class _DetectPatternOutput:
    input_: _DetectPatternInput
    pattern_length_logic: _PatternLengthLogic
    pattern_start_logic: _PatternStartLogic
    pattern: _Pattern | None
    stats: tuple[
        _Duration,
        _SkipAllReason
        | dict[
            _PatternLength,
            tuple[
                _Duration,
                _PatternLengthSkipReason
                | dict[
                    _PatternIndex,
                    tuple[_Duration, _PatternLengthAndStartNotFoundReason | None],
                ],
            ],
        ],
    ]
    processed_max_length: _PatternLength | None
    processed_max_starts: dict[_PatternLength, _PatternIndex]

    @property
    def module_name(self) -> _ModuleName:
        return self.input_.module_name


def _determine_lengths_to_process(
    *,
    pattern_length_logic: _PatternLengthLogic,
    pattern_start_logic: _PatternStartLogic,
    check_type: _CheckType,
    max_length: _PatternLength,
    previously_processed_max_length: _PatternLength | None,
) -> Iterator[_PatternLength]:
    if check_type == "full":
        lengths = range(2, max_length + 1)
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

            if (
                pattern_start_logic is _PatternStartLogic.AtZero
                and previously_processed_max_length is not None
            ):
                it = itertools.dropwhile(
                    lambda length: length < previously_processed_max_length, it
                )

            lengths = itertools.takewhile(lambda length: length <= max_length, it)

        elif pattern_length_logic is _PatternLengthLogic.Any:
            if pattern_start_logic is _PatternStartLogic.AtZero:
                min_sublist_length = max(2, (previously_processed_max_length or 0) + 1)
            elif pattern_start_logic is _PatternStartLogic.Anywhere:
                min_sublist_length = 2
            else:
                assert_never(pattern_start_logic)
            lengths = range(min_sublist_length, max_length + 1)
        else:
            assert_never(pattern_length_logic)

    yield from map(_PatternLength, lengths)


def _determine_start_indices_to_process(
    *,
    pattern_start_logic: _PatternStartLogic,
    check_type: _CheckType,
    length: _PatternLength,
    signals: list[list[_Record]],
    previously_processed_max_starts: dict[_PatternLength, _PatternIndex],
) -> Iterable[_PatternIndex]:
    if check_type == "full":
        min_sublist1_start = _PatternIndex(0)
        max_sublist1_start = _PatternIndex(len(signals) - 2 * length)
    elif pattern_start_logic is _PatternStartLogic.AtZero:
        min_sublist1_start = _PatternIndex(0)
        max_sublist1_start = _PatternIndex(0)
    elif pattern_start_logic is _PatternStartLogic.Anywhere:
        min_sublist1_start = _PatternIndex(
            previously_processed_max_starts.get(length, -1) + 1
        )
        max_sublist1_start = _PatternIndex(len(signals) - 2 * length)
    else:
        assert_never(pattern_start_logic)

    yield from map(
        _PatternIndex,
        range(min_sublist1_start, max_sublist1_start + 1),
    )


def _detect_pattern_for_length_and_start(
    *,
    length: _PatternLength,
    start: _PatternIndex,
    check_type: _CheckType,
    signals: list[list[_Record]],
    required_input_signals: Counter[tuple[_ModuleName, _Signal]],
) -> _PatternLengthAndStartNotFoundReason | _Pattern:
    end = start + length
    sublist = signals[start:end]

    # Check that all required sources are present in sublist
    contained_inputs = Counter[tuple[_ModuleName, _Signal]]()
    for record in itertools.chain(*sublist):
        contained_inputs.update(((record.source, record.source_signal),))
        if required_input_signals == contained_inputs & required_input_signals:
            break
    else:
        return "missing_required_inputs"

    remaining_sublists_count = (len(signals) - end) // length
    assert remaining_sublists_count >= 1
    if check_type == "minimal":
        # During normal execution, only check first following sublist
        remaining_sublists_count = 1

    all_sublists_match = True
    for start_other in map(
        _PatternIndex,
        range(end, end + remaining_sublists_count * length, length),
    ):
        end_other = _PatternIndex(start_other + length)
        assert end_other <= len(signals)
        sublist_other = signals[start_other:end_other]

        # Check that sublist1 and sublist_other have same contents
        if sublist != sublist_other:
            all_sublists_match = False
            break

    if not all_sublists_match:
        return "following_sublist_mismatch"

    if start != 0:
        raise AssertionError("Proof that start isn't always 0")  # noqa: TRY003
    if len(sublist) > pre_calculated_twos_exponents_for_lengths[-1]:
        raise AssertionError(  # noqa: TRY003
            "Need to pre calculate more 2**x lenghts"
        )
    if len(sublist) not in pre_calculated_twos_exponents_for_lengths:
        raise AssertionError(  # noqa: TRY003
            "Proof that pattern length isn't always 2**x"
        )

    return _Pattern(start, sublist)


def _detect_pattern_for_length(
    *,
    pattern_start_logic: _PatternStartLogic,
    check_type: _CheckType,
    length: _PatternLength,
    required_input_signals: Counter[tuple[_ModuleName, _Signal]],
    signals: list[list[_Record]],
    previously_processed_max_starts: dict[_PatternLength, _PatternIndex],
) -> tuple[
    _PatternIndex,
    _PatternLengthSkipReason
    | tuple[
        _Pattern | None,
        dict[
            _PatternIndex,
            tuple[_Duration, _PatternLengthAndStartNotFoundReason | None],
        ],
    ],
]:
    starts = list(
        _determine_start_indices_to_process(
            pattern_start_logic=pattern_start_logic,
            check_type=check_type,
            length=length,
            signals=signals,
            previously_processed_max_starts=previously_processed_max_starts,
        )
    )

    min_start = starts[0]
    max_start = starts[-1]
    # Check that all required sources are present in signals[min, max]
    if min_start > 0:  # If 0, then this has been checked already for all signals
        contained_inputs = Counter[tuple[_ModuleName, _Signal]]()
        for record in itertools.chain(*signals[min_start : max_start + length]):
            contained_inputs.update(((record.source, record.source_signal),))
            if required_input_signals == contained_inputs & required_input_signals:
                break
        else:
            return max_start, "missing_required_inputs"

    stats: dict[
        _PatternIndex,
        tuple[_Duration, _PatternLengthAndStartNotFoundReason | None],
    ] = {}

    pattern: _Pattern | None = None
    for start in starts:
        time_begin = time.perf_counter()
        start_result = _detect_pattern_for_length_and_start(
            length=length,
            start=start,
            check_type=check_type,
            signals=signals,
            required_input_signals=required_input_signals,
        )
        if isinstance(start_result, _Pattern):
            pattern = start_result
            start_stat_data = None
        else:
            start_stat_data = start_result
        stats[start] = (
            _Duration(time.perf_counter() - time_begin),
            start_stat_data,
        )
        if pattern is not None:
            break

    return max_start, (pattern, stats)


def _detect_pattern(
    pattern_length_logic: _PatternLengthLogic,
    pattern_start_logic: _PatternStartLogic,
    input_: _DetectPatternInput,
) -> _DetectPatternOutput:
    time_begin_task = time.perf_counter()

    # Check that all required sources are present in signals
    contained_inputs = Counter[tuple[_ModuleName, _Signal]]()
    for record in itertools.chain(*input_.signals):
        contained_inputs.update(((record.source, record.source_signal),))
        if (
            input_.required_input_signals
            == contained_inputs & input_.required_input_signals
        ):
            break
    else:
        return _DetectPatternOutput(
            input_=input_,
            pattern_length_logic=pattern_length_logic,
            pattern_start_logic=pattern_start_logic,
            pattern=None,
            stats=(
                _Duration(time.perf_counter() - time_begin_task),
                "missing_required_inputs",
            ),
            processed_max_length=None,
            processed_max_starts={},
        )

    check_type = "full"
    previously_processed_max_length = None
    previously_processed_max_starts = {}
    if isinstance(input_, _DetectPatternInputMinimal):
        check_type = "minimal"
        previously_processed_max_length = input_.previously_processed_max_length
        previously_processed_max_starts = input_.previously_processed_max_starts

    max_pattern_length = _PatternLength(len(input_.signals) // 2)
    lengths = list(
        _determine_lengths_to_process(
            pattern_length_logic=pattern_length_logic,
            pattern_start_logic=pattern_start_logic,
            check_type=check_type,
            max_length=max_pattern_length,
            previously_processed_max_length=previously_processed_max_length,
        )
    )

    stats: dict[
        _PatternLength,
        tuple[
            _Duration,
            _PatternLengthSkipReason
            | dict[
                _PatternIndex,
                tuple[_Duration, _PatternLengthAndStartNotFoundReason | None],
            ],
        ],
    ] = {}
    pattern: _Pattern | None = None
    processed_max_starts: dict[_PatternLength, _PatternIndex] = {}
    for length in lengths:
        time_begin = time.perf_counter()
        processed_max_start, result = _detect_pattern_for_length(
            pattern_start_logic=pattern_start_logic,
            check_type=check_type,
            length=length,
            required_input_signals=input_.required_input_signals,
            signals=input_.signals,
            previously_processed_max_starts=previously_processed_max_starts,
        )
        processed_max_starts[length] = processed_max_start
        if not isinstance(result, tuple):
            stats[length] = (
                _Duration(time.perf_counter() - time_begin),
                result,
            )
            continue

        pattern, stats_starts = result
        stats[length] = (
            _Duration(time.perf_counter() - time_begin),
            stats_starts,
        )
        if pattern is not None:
            break

    return _DetectPatternOutput(
        input_=input_,
        pattern_length_logic=pattern_length_logic,
        pattern_start_logic=pattern_start_logic,
        pattern=pattern,
        stats=(
            _Duration(time.perf_counter() - time_begin_task),
            stats,
        ),
        processed_max_length=max(stats.keys()),
        processed_max_starts=processed_max_starts,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class _PatternLogicConfig:
    analysis: _PatternAnalysisLogic
    length: _PatternLengthLogic
    start: _PatternStartLogic


class _PatternDetector:
    def __init__(
        self,
        module_name: _ModuleName,
        required_inputs: frozenset[_ModuleName],
        required_signals: frozenset[_Signal],
        required_signals_count: int,
        pattern_logic_config: _PatternLogicConfig,
    ) -> None:
        self.module_name = module_name
        self._required_input_signals = Counter(
            {
                input_: required_signals_count
                for input_ in set(itertools.product(required_inputs, required_signals))
            }
        )
        self._pattern_logic_config = pattern_logic_config

        self._signals: list[list[_Record]] = []

        self._detected_pattern: _Pattern | None = None

        # Caches for minimal check type tasks
        self._previous_max_start_indices: dict[_PatternLength, _PatternIndex] = {}
        self._previous_max_length: _PatternLength | None = None

    @property
    def detected_pattern(self) -> _Pattern | None:
        return self._detected_pattern

    @property
    def signals(self) -> list[list[_Record]]:
        return self._signals

    def create_detect_pattern_task(self, check_type: _CheckType) -> _DetectPatternInput:
        if check_type == "minimal":
            assert self._detected_pattern is None
        elif check_type == "full":
            assert self._detected_pattern is not None
        else:
            assert_never(check_type)
        return self._create_detect_pattern_task_input(
            check_type=check_type,
            latest_button_press_completed=True,
        )

    def process_detect_pattern_task_output(self, output: _DetectPatternOutput) -> None:
        if output.module_name != self.module_name:
            raise ValueError(output.input_.module_name)

        self._detected_pattern = output.pattern

        if isinstance(output.input_, _DetectPatternInputMinimal):
            self._previous_max_length = output.processed_max_length
            for length, processed_max_start in output.processed_max_starts.items():
                previous_max_start = self._previous_max_start_indices.get(length)
                if (
                    previous_max_start is not None
                    and previous_max_start > processed_max_start
                ):
                    raise AssertionError()
            self._previous_max_start_indices = output.processed_max_starts

    def _create_detect_pattern_task_input(
        self, check_type: _CheckType, latest_button_press_completed: bool
    ) -> _DetectPatternInput:
        signals = self._signals if latest_button_press_completed else self._signals[:-1]
        match check_type:
            case "full":
                return _DetectPatternInputFull(
                    module_name=self.module_name,
                    required_input_signals=self._required_input_signals,
                    signals=signals,
                )
            case "minimal":
                return _DetectPatternInputMinimal(
                    module_name=self.module_name,
                    required_input_signals=self._required_input_signals,
                    signals=signals,
                    previously_processed_max_length=self._previous_max_length,
                    previously_processed_max_starts=self._previous_max_start_indices,
                )

    def record_processed_input(
        self,
        button_press_count: int,
        source: _ModuleName,
        source_signal: _Signal,
        resulting_signal: _Signal | None,
    ) -> None:
        assert len(self._signals) <= button_press_count
        appended = False
        while len(self._signals) < button_press_count:
            self._signals.append([])
            appended = True
        self._signals[-1].append(_Record(source, source_signal, resulting_signal))

        if (
            self._pattern_logic_config.analysis
            is _PatternAnalysisLogic.AllNewButtonPressesPerModule
            and self.detected_pattern is None
            and appended is True
        ):
            self.process_detect_pattern_task_output(
                _detect_pattern(
                    self._pattern_logic_config.length,
                    self._pattern_logic_config.start,
                    self._create_detect_pattern_task_input(
                        check_type="minimal", latest_button_press_completed=False
                    ),
                )
            )


class _Module:
    def __init__(self, name: _ModuleName) -> None:
        self._name = name
        self._pattern_detector: _PatternDetector | None = None

    @property
    def name(self) -> _ModuleName:
        return self._name

    @property
    def pattern_detector(self) -> _PatternDetector:
        # Only requested when it exists: p1 doesn't use pattern detector and doesn't
        # access this, p2 uses and accesses.
        assert self._pattern_detector is not None
        return self._pattern_detector

    @final
    def process_input(
        self, button_press_count: int, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        result = self._process_input_internal(source_name, state)
        if self._pattern_detector is not None:
            self._pattern_detector.record_processed_input(
                button_press_count, source_name, state, result
            )
        return result

    def _process_input_internal(
        self, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._name})"


class _FlipFlop(_Module):
    def __init__(self, name: _ModuleName) -> None:
        super().__init__(name=name)
        self._state: _Signal = "low"

    def create_pattern_detector(
        self,
        inputs: Iterable[_ModuleName],
        pattern_logic_config: _PatternLogicConfig,
    ) -> None:
        self._pattern_detector = _PatternDetector(
            self._name, frozenset(inputs), frozenset(("low",)), 2, pattern_logic_config
        )

    @override
    def _process_input_internal(
        self, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        if state == "high":
            return None

        match self._state:
            case "low":
                self._state = "high"
            case "high":
                self._state = "low"
        return self._state

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._name}, {self._state})"


class _Conjuction(_Module):
    def __init__(self, name: _ModuleName) -> None:
        super().__init__(name=name)
        self._state: dict[_ModuleName, _Signal] = {}

    def create_pattern_detector(
        self,
        inputs: Iterable[_ModuleName],
        pattern_logic_config: _PatternLogicConfig,
    ) -> None:
        self._pattern_detector = _PatternDetector(
            self._name,
            frozenset(inputs),
            frozenset(
                ("low", "high"),
            ),
            1,
            pattern_logic_config,
        )

    def set_inputs(self, inputs: Iterable[_ModuleName]) -> None:
        self._state.update({name: "low" for name in inputs})

    @override
    def _process_input_internal(
        self, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        assert source_name in self._state
        self._state[source_name] = state
        return (
            "low" if all(value == "high" for value in self._state.values()) else "high"
        )

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._name}, {self._state})"


class _Broadcast(_Module):
    def __init__(self, name: _ModuleName) -> None:
        super().__init__(name=name)

    def create_pattern_detector(
        self, inputs: Iterable[_ModuleName], pattern_logic_config: _PatternLogicConfig
    ) -> None:
        inputs = list(inputs)
        assert len(inputs) == 0
        self._pattern_detector = _PatternDetector(
            self._name,
            frozenset((_ModuleName(""),)),
            frozenset(("low",)),
            1,
            pattern_logic_config,
        )

    @override
    def _process_input_internal(
        self, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        return state


class _Receiver(_Module):
    def __init__(self, name: _ModuleName) -> None:
        super().__init__(name=name)

    def create_pattern_detector(
        self, inputs: Iterable[_ModuleName], pattern_logic_config: _PatternLogicConfig
    ) -> None:
        self._pattern_detector = _PatternDetector(
            self._name, frozenset(inputs), frozenset(["low"]), 1, pattern_logic_config
        )

    @override
    def _process_input_internal(
        self, source_name: _ModuleName, state: _Signal
    ) -> _Signal | None:
        if state == "low":
            _logger.warning("Receiver received signal from %s: %s", source_name, state)
        return None


type AnyModule = _FlipFlop | _Conjuction | _Broadcast | _Receiver


def _parse_module(line: str) -> tuple[AnyModule, list[_ModuleName]]:
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
    return _Conjuction(name), outputs


def _parse_modules(
    lines: Iterable[str],
    pattern_logic_config: _PatternLogicConfig | None = None,
) -> digraph.Digraph[_ModuleName, AnyModule]:
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
        inputs = list(map(lambda arc: arc.from_, graph.get_arcs_to(name)))
        if isinstance(module, _Conjuction):
            module.set_inputs(inputs)
        if pattern_logic_config is not None:
            module.create_pattern_detector(inputs, pattern_logic_config)

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

    pattern_logic_config = _PatternLogicConfig(
        analysis=_PatternAnalysisLogic.AllModulesInBatches,
        length=_PatternLengthLogic.Exp2,
        start=_PatternStartLogic.AtZero,
    )
    post_process_logic = PostProcessLogic.InBatches
    batch_length = 1_000
    pattern_analysis_logic = pattern_logic_config.analysis

    graph = _parse_modules(input_str.splitlines(), pattern_logic_config)

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

    def log_task_output(task_output: _DetectPatternOutput) -> None:
        duration = task_output.stats[0]
        if isinstance(task_output.stats[1], dict):
            desc = "No pattern found"
        else:
            desc = task_output.stats[1]
        _logger.info("%s: %s seconds, %s", task_output.module_name, duration, desc)

    def process_pattern_detection_for_nodes(
        nodes: Iterable[_ModuleName],
        check_type: _CheckType,
    ) -> list[_DetectPatternOutput]:
        task_inputs = [
            pattern_detector.create_detect_pattern_task(check_type)
            for pattern_detector in map(
                lambda node: graph.nodes[node].pattern_detector, nodes
            )
        ]

        def run_detection_task(task_input: _DetectPatternInput) -> _DetectPatternOutput:
            _logger.debug("Running detection task for %s", task_input.module_name)
            return _detect_pattern(
                pattern_logic_config.length,
                pattern_logic_config.start,
                task_input,
            )

        task_outputs = [run_detection_task(task_input) for task_input in task_inputs]

        for task_output in task_outputs:
            log_task_output(task_output)
            pattern_detector = graph.nodes[task_output.module_name].pattern_detector
            pattern_detector.process_detect_pattern_task_output(task_output)

        return task_outputs

    def detect_individual_patterns(
        nodes: set[_ModuleName],
    ) -> tuple[set[_ModuleName], list[_DetectPatternOutput]]:
        nodes_to_process = sorted(
            nodes - nodes_with_finalized_pattern - nodes_with_considered_pattern
        )

        minimal_pattern_outputs = list[_DetectPatternOutput]()
        if pattern_analysis_logic is _PatternAnalysisLogic.AllModulesInBatches:
            minimal_pattern_outputs = process_pattern_detection_for_nodes(
                nodes_to_process, "minimal"
            )

        nodes_detected_with_considered_pattern = set[_ModuleName]()
        nodes_detected_for_final_check = list[_ModuleName]()
        for node in nodes_to_process:
            module = graph.nodes[node]
            detector = module.pattern_detector
            if detector.detected_pattern is None:
                continue

            if node in map_of_nodes_to_nodes_in_same_cycle:
                nodes_detected_with_considered_pattern.add(node)
                continue

            assert node in nodes_not_in_cycles
            from_nodes = set(map(lambda arc: arc.from_, graph.get_arcs_to(node)))
            if not from_nodes <= nodes_with_finalized_pattern:
                continue

            nodes_detected_for_final_check.append(node)

        process_pattern_detection_for_nodes(
            nodes_detected_for_final_check,
            "full",
        )
        for node in nodes_detected_for_final_check:
            detector = graph.nodes[node].pattern_detector
            assert detector.detected_pattern is not None, "Not supported scenario"
            nodes_with_finalized_pattern.add(node)

        return nodes_detected_with_considered_pattern, minimal_pattern_outputs

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

    assert not graph.get_arcs_from(_ModuleName("rx"))
    assert list(map(lambda x: x.from_, graph.get_arcs_to(_ModuleName("rx")))) == [
        _ModuleName("kh")
    ]
    signals_to_kh = list[tuple[_ModuleName, _Signal]]()

    for button_press_count in itertools.count(1):
        if (
            _logger.isEnabledFor(logging.INFO)
            and button_press_count % batch_length == 0
        ):
            _logger.info("Button pressed. Count: %d", button_press_count)
        result = _process_button_press(
            _ButtonPressProcessInput(graph, button_press_count)
        )

        signals_to_kh.extend(
            (signal.source, signal.state)
            for signal in result.signals
            if signal.destination == _ModuleName("kh")
        )

        if post_process_logic is PostProcessLogic.OnEveryButtonPress:
            destinations = {signal.destination for signal in result.signals}
            nodes_detected_with_considered_pattern, minimal_pattern_outputs = (
                detect_individual_patterns(destinations)
            )
        elif post_process_logic is PostProcessLogic.InBatches:
            if button_press_count % batch_length != 0:
                continue

            nodes = set(graph.nodes)
            nodes_detected_with_considered_pattern, minimal_pattern_outputs = (
                detect_individual_patterns(nodes)
            )
        else:
            assert_never(post_process_logic)

        nodes_with_considered_pattern |= nodes_detected_with_considered_pattern
        nodes_with_some_pattern = (
            nodes_with_finalized_pattern | nodes_with_considered_pattern
        )

        nodes_to_process = set(nodes_detected_with_considered_pattern)
        nodes_detected_for_final_check = set[_ModuleName]()
        while nodes_to_process:
            node = nodes_to_process.pop()
            detector = graph.nodes[node].pattern_detector

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

            nodes_detected_for_final_check |= (
                nodes_in_shared_cycles - nodes_with_finalized_pattern
            )

        process_pattern_detection_for_nodes(
            sorted(nodes_detected_for_final_check), "full"
        )

        for node in nodes_detected_for_final_check:
            detector = graph.nodes[node].pattern_detector
            assert detector.detected_pattern is not None, "Not supported scenario"

        nodes_with_finalized_pattern |= nodes_detected_for_final_check
        nodes_with_considered_pattern -= nodes_with_finalized_pattern

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

        if post_process_logic is PostProcessLogic.OnEveryButtonPress:
            continue

        conjuction_modules = {
            module.name: module
            for module in graph.nodes.values()
            if isinstance(module, _Conjuction)
        }
        conjuction_outputs = [
            (module, output)
            for output in minimal_pattern_outputs
            if (module := conjuction_modules.get(output.module_name)) is not None
        ]
        not_all_resolved = False
        for _, output in conjuction_outputs:
            assert isinstance(output.input_, _DetectPatternInputMinimal)
            if (
                output.input_.previously_processed_max_length is None
                and output.processed_max_length is None
            ):
                not_all_resolved = True
                break
        if not_all_resolved:
            _logger.debug("Not all Conjuctions have all required inputs")
            continue

        detected_any = False
        for module, _ in conjuction_outputs:
            press_counts = list[int]()
            for press_index, button_press_signals in enumerate(
                module.pattern_detector.signals
            ):
                low_inputs = set[_ModuleName]()
                high_inputs = set[_ModuleName]()
                for signal in button_press_signals:
                    if signal.source_signal == "low":
                        low_inputs.add(signal.source)
                    else:
                        assert signal.source_signal == "high"
                        high_inputs.add(signal.source)
                if low_inputs & high_inputs:
                    press_counts.append(press_index + 1)
            if press_counts:
                detected_any = True
                _logger.warning(
                    "Both high and low signals sent to '%s' during one button press %d",
                    module.name,
                    press_counts,
                )
        if not detected_any:
            _logger.info(
                "No high and low signals sent from any input to Conjunction during "
                "one button press"
            )

    return button_press_count
