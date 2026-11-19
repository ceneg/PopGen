"""Run context: the provenance log, fatal errors and memory checks."""

import logging
import os
import platform
import shlex
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import NoReturn

import psutil

import popgen

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Run from a checkout (scripts/ next to the package): logs go to <checkout>/logs;
# when installed elsewhere, to ./logs
_IN_CHECKOUT = (PROJECT_ROOT / "scripts").is_dir()
LOG_PACKAGES = ("numpy", "pandas", "scipy", "scikit-allel", "scikit-learn",
                "biopython", "networkx", "statsmodels", "matplotlib", "seaborn")


def _log_uncaught(exc_type, exc, tb) -> None:
    """``sys.excepthook``: record the traceback in the log, then the default hook."""
    logging.getLogger().critical("Uncaught exception", exc_info=(exc_type, exc, tb))
    sys.__excepthook__(exc_type, exc, tb)


def setup_logging(script: str, args=None, log_dir: str | Path | None = None) -> Path:
    """
    Send log messages to the console and to ``logs/<script>_<timestamp>.log``,
    and record the run details in the log file.

    Warnings (``warnings.warn``) and uncaught exceptions are logged too, so the
    log of a failed run ends with the reason.

    Args:
        script: Script name used for the log file name.
        args: Parsed ``argparse`` namespace to record, or None.
        log_dir: Directory for the log file (default: the ``POPGEN_LOG_DIR``
            environment variable if set, otherwise ``<checkout>/logs``, or
            ``./logs`` when the package is installed outside a checkout).

    Returns:
        Path: The log file.
    """
    log_dir = log_dir or os.environ.get("POPGEN_LOG_DIR")
    if not log_dir:
        log_dir = PROJECT_ROOT / "logs" if _IN_CHECKOUT else Path.cwd() / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script}_{stamp}_{os.getpid()}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    to_file = logging.FileHandler(log_file)
    to_file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(console)
    root.addHandler(to_file)
    logging.captureWarnings(True)
    sys.excepthook = _log_uncaught

    versions = []
    for pkg in LOG_PACKAGES:
        try:
            versions.append(f"{pkg}={importlib_metadata.version(pkg)}")
        except importlib_metadata.PackageNotFoundError:
            pass
    # Run details go to the log file only
    for line in (
        f"PopGen {popgen.__version__}",
        f"Command: {shlex.join(sys.argv)}",
        f"Working directory: {Path.cwd()}",
        f"Parameters: {vars(args) if args is not None else {}}",
        f"Python {platform.python_version()}; " + ", ".join(versions),
    ):
        to_file.handle(
            logging.LogRecord(script, logging.INFO, __file__, 0, line, None, None)
        )
    return log_file


def die(message: str, code: int = 1) -> NoReturn:
    """
    Log a fatal error (console and log file) and exit.

    Args:
        message: What went wrong, without an "Error:" prefix.
        code: Exit status.

    Raises:
        SystemExit: Always.
    """
    logging.getLogger("popgen").error("Error: " + message)
    raise SystemExit(code)


def check_memory(required_bytes: float, what: str) -> bool:
    """
    Warn when ``required_bytes`` exceeds the memory currently available.

    Args:
        required_bytes: Estimated peak memory of the next step.
        what: Description of the step, for the message.

    Returns:
        bool: True if enough memory appears to be available.
    """
    available = psutil.virtual_memory().available
    if required_bytes > available:
        log.warning(
            f"Warning: {what} may need ~{required_bytes / 1e9:.1f} GB, "
            f"but only {available / 1e9:.1f} GB of memory is available."
        )
        return False
    return True
