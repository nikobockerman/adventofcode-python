from adventofcode.tooling.ranges import partition_range


def test_partition_range() -> None:
    assert partition_range(range(5, 10), range(4, 5)) == (
        range(0),
        range(0),
        range(5, 10),
    )
    assert partition_range(range(5, 10), range(3, 8)) == (
        range(0),
        range(5, 8),
        range(8, 10),
    )

    assert partition_range(range(5, 10), range(3, 12)) == (
        range(0),
        range(5, 10),
        range(0),
    )

    assert partition_range(range(5, 10), range(8, 12)) == (
        range(5, 8),
        range(8, 10),
        range(0),
    )

    assert partition_range(range(5, 10), range(7, 9)) == (
        range(5, 7),
        range(7, 9),
        range(9, 10),
    )
