"""Compare in-memory projections with JSON-persisted event projections."""

import pytest


@pytest.fixture
def assert_rebuilt():
    def compare(actual, expected):
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                compare(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                compare(a, b)
        elif isinstance(expected, float):
            # Rust's JSON log reader can differ by one ULP from the live value.
            assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)
        else:
            assert actual == expected

    return lambda pit: compare(pit.rebuild(), (pit.snapshot(), pit.active))
