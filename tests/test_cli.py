import argparse

import pytest

from popgen.cli import fraction, non_negative_int, positive_float, positive_int


@pytest.mark.parametrize("parse, text", [
    (positive_int, "0"),
    (positive_int, "-3"),
    (positive_int, "1.5"),
    (non_negative_int, "-1"),
    (positive_float, "0"),
    (positive_float, "inf"),
    (positive_float, "nan"),
    (fraction, "0"),
    (fraction, "1.5"),
    (fraction, "x"),
])
def test_rejected(parse, text):
    """T-CLI-1 (B15)."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse(text)


@pytest.mark.parametrize("parse, text, value", [
    (positive_int, "3", 3),
    (non_negative_int, "0", 0),
    (positive_float, "0.05", 0.05),
    (fraction, "1", 1.0),
    (fraction, "0.4", 0.4),
])
def test_accepted(parse, text, value):
    assert parse(text) == value
