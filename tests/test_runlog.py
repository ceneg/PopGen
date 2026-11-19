import argparse
import logging
import sys
import warnings

import pytest

import popgen
from conftest import PROJECT_ROOT, run_script
from popgen.runlog import die, setup_logging


@pytest.fixture
def clean_logging():
    """Undo setup_logging: root handlers, warning capture and excepthook."""
    logging.captureWarnings(False)
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    logging.captureWarnings(False)
    sys.excepthook = sys.__excepthook__


def flush():
    for h in logging.getLogger().handlers:
        h.flush()


def test_setup_logging_writes_run_details(tmp_path, clean_logging):
    log_file = setup_logging("demo", argparse.Namespace(vcf="x.vcf"), tmp_path)
    logging.getLogger("demo").info("hello")
    flush()
    text = log_file.read_text()
    assert "Parameters: {'vcf': 'x.vcf'}" in text
    assert "numpy=" in text and "hello" in text
    assert f"PopGen {popgen.__version__}" in text
    assert "Working directory: " in text and "Command: " in text


def test_setup_logging_closes_replaced_handlers(tmp_path, clean_logging):
    setup_logging("demo", None, tmp_path / "a")
    old = [h for h in logging.getLogger().handlers
           if isinstance(h, logging.FileHandler)]
    setup_logging("demo", None, tmp_path / "b")
    assert old and all(h.stream is None for h in old)  # closed


def test_die_and_warnings_reach_log(tmp_path, clean_logging):
    """T-LOG-1 (B13)."""
    log_file = setup_logging("demo", None, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("careful")
    with pytest.raises(SystemExit) as exc:
        die("boom")
    assert exc.value.code == 1
    flush()
    text = log_file.read_text()
    assert "Error: boom" in text and "careful" in text


def test_fatal_error_is_in_log_file(tmp_path):
    """T-LOG-2 (B13): the log of a failed run ends with the reason."""
    data = PROJECT_ROOT / "examples" / "data"
    result = run_script("structure", tmp_path, "--vcf", tmp_path / "missing.vcf",
                        "--meta", data / "metadata.tsv", "--out", tmp_path / "ex")
    assert result.returncode != 0
    logs = list((tmp_path / "logs").iterdir())
    assert len(logs) == 1
    assert "missing.vcf" in logs[0].read_text().splitlines()[-1]


def test_uncaught_exception_is_logged(tmp_path, clean_logging):
    log_file = setup_logging("demo", None, tmp_path)
    try:
        raise ValueError("unexpected")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    flush()
    text = log_file.read_text()
    assert "CRITICAL" in text and "ValueError: unexpected" in text
