from adventofcode.tooling import debugger


def test_no_debugger():
    assert debugger.is_connected() is False
