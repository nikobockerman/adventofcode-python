import pytest
from pytest_benchmark.fixture import BenchmarkFixture


class A:
    __slots__ = ("x",)

    def __init__(self):
        self.x = 1


class B:
    __slots__ = ("_x",)

    def __init__(self):
        self._x = 1

    @property
    def x(self) -> int:
        return self._x


def compare(inst: A | B) -> None:
    assert inst.x == 1


@pytest.fixture(params=[A, B])
def instance(request: pytest.FixtureRequest) -> A | B:
    return request.param()


def test_benchmark(benchmark: BenchmarkFixture, instance: A | B) -> None:
    benchmark(compare, instance)
