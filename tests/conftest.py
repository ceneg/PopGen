import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load(name):
    return importlib.import_module(f"popgen.analyses.{name}")


@pytest.fixture(scope="session")
def load_script():
    """Import an analysis module (``popgen.analyses.<name>``) by name."""
    return _load


def run_script(script, tmp, *args, cwd=None):
    """Run scripts/<script>.py with logs in tmp/logs; return the CompletedProcess."""
    import os
    import subprocess

    env = dict(os.environ, POPGEN_LOG_DIR=str(Path(tmp) / "logs"))
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / f"{script}.py"), *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)


def example_vcf_subset(path, n_samples):
    """Write examples/data/example.vcf with only its first n_samples samples."""
    lines = []
    src = PROJECT_ROOT / "examples" / "data" / "example.vcf"
    for line in src.read_text().splitlines():
        if not line.startswith("##"):
            line = "\t".join(line.split("\t")[: 9 + n_samples])
        lines.append(line)
    Path(path).write_text("\n".join(lines) + "\n")
    return path
