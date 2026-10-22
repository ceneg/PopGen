import subprocess
import sys

from conftest import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "generate_citations.py"


def test_help_writes_nothing(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "-h"], cwd=tmp_path, capture_output=True
    )
    assert result.returncode == 0
    assert not any(tmp_path.iterdir())


def test_report_lists_script_dependencies(tmp_path):
    out = tmp_path / "versions.txt"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    text = out.read_text()
    assert "Script: selection.py" in text and "- Bio: " in text
