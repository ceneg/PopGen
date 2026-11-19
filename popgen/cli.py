"""Command-line helpers: checked ``argparse`` types and common arguments."""

import argparse


def _number(text: str, kind):
    try:
        return kind(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid {kind.__name__}: {text!r}")


def positive_int(text: str) -> int:
    """``argparse`` type: an integer > 0."""
    value = _number(text, int)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def non_negative_int(text: str) -> int:
    """``argparse`` type: an integer >= 0."""
    value = _number(text, int)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text}")
    return value


def positive_float(text: str) -> float:
    """``argparse`` type: a finite number > 0."""
    value = _number(text, float)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError(f"must be a number > 0, got {text}")
    return value


def fraction(text: str) -> float:
    """``argparse`` type: a number in (0, 1]."""
    value = _number(text, float)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"must be > 0 and <= 1, got {text}")
    return value


def add_out_argument(
    parser: argparse.ArgumentParser, required: bool = True, default: str | None = None
) -> None:
    """Add ``--out DIR/NAME`` (outputs go to ``DIR/results/NAME.*``)."""
    parser.add_argument(
        "--out",
        required=required,
        default=default,
        help="Output prefix DIR/NAME: files are written to DIR/results/NAME.*"
        + (f" (default {default})" if default else ""),
    )
