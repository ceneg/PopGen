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


def test_default_prints_and_writes_nothing(tmp_path):
    """T-CIT-1 (B20): without --out, the report goes to stdout only."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Script: selection.py" in result.stdout
    assert not any(tmp_path.iterdir())


def test_out_writes_same_report(tmp_path):
    """T-CIT-2 (R9): --out writes the printed text; imports are followed
    through popgen.* modules."""
    printed = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True,
        check=True).stdout
    out = tmp_path / "sub" / "f.txt"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)], cwd=tmp_path,
        capture_output=True, text=True, check=True)
    text = out.read_text()
    assert text == printed and result.stdout == ""
    assert str(out) in result.stderr
    sections = {s.split("\n", 1)[0]: s for s in text.split("Script: ")[1:]}
    assert set(sections) == {f"{n}.py" for n in (
        "diversity", "ld_decay", "network", "recombination", "selection", "structure")}
    assert "- Bio: " in sections["selection.py"]
    assert "- networkx: " in sections["network.py"]
    assert "- allel: " in sections["diversity.py"]  # only via popgen.io
    assert "- Phi: " in sections["recombination.py"]
