import hashlib
import importlib.util
import io
import shutil
import subprocess
import tarfile

import pytest

from conftest import PROJECT_ROOT


@pytest.fixture(scope="module")
def inst():
    path = PROJECT_ROOT / "tools" / "install_phipack.py"
    spec = importlib.util.spec_from_file_location("install_phipack", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAIN_C = b'#include <stdio.h>\nint main(void) { puts("Usage: Phi"); return 0; }\n'


def fake_archive(inst):
    """PhiPack-like tar.gz whose 'Phi' prints a usage line (plus macOS ._ files)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in inst.PHI_SOURCES:
            code = MAIN_C if name == "main.c" else b""
            for path in (f"PhiPack/src/{name}", f"PhiPack/src/._{name}"):
                info = tarfile.TarInfo(path)
                info.size = len(code)
                tar.addfile(info, io.BytesIO(code))
    return buf.getvalue()


def test_verify_rejects_wrong_checksum(inst):
    with pytest.raises(ValueError, match="checksum mismatch"):
        inst.verify(b"not phipack")


def test_find_source_dir_requires_all_sources(inst, tmp_path):
    src = tmp_path / "PhiPack" / "src"
    src.mkdir(parents=True)
    (src / "main.c").write_text("")
    with pytest.raises(FileNotFoundError):
        inst.find_source_dir(tmp_path)
    for name in inst.PHI_SOURCES:
        (src / name).write_text("")
    assert inst.find_source_dir(tmp_path) == src


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_install_builds_and_records_version(inst, tmp_path):
    archive = fake_archive(inst)
    sha = hashlib.sha256(archive).hexdigest()
    phi = inst.install(archive, tmp_path / "bin", "cc", sha256=sha)
    inst.check(phi)
    assert "Usage" in subprocess.run([phi], capture_output=True, text=True).stdout
    assert (tmp_path / "bin" / inst.VERSION_FILE).read_text().startswith("PhiPack 1.1")


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_tries_next_url_after_incomplete_read(inst):
    """T-PHI-1 (B21 e): IncompleteRead is not an OSError; the next URL is tried."""
    import http.client

    data = b"archive"
    calls = []

    def opener(url, timeout):
        calls.append(url)
        if url == "first":
            raise http.client.IncompleteRead(b"part")
        return FakeResponse(data)

    sha = hashlib.sha256(data).hexdigest()
    assert inst.download(("first", "second"), sha, opener=opener) == data
    assert calls == ["first", "second"]


def test_install_refuses_foreign_phi(inst, tmp_path):
    """T-PHI-1 (B21 f): a Phi this tool did not install is not overwritten."""
    dest = tmp_path / "bin"
    dest.mkdir()
    (dest / "Phi").write_text("conda's Phi")
    archive = fake_archive(inst)
    sha = hashlib.sha256(archive).hexdigest()
    with pytest.raises(RuntimeError, match="--force"):
        inst.install(archive, dest, "cc", sha256=sha)
    assert (dest / "Phi").read_text() == "conda's Phi"


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_install_replaces_own_or_forced_phi(inst, tmp_path):
    archive = fake_archive(inst)
    sha = hashlib.sha256(archive).hexdigest()
    dest = tmp_path / "bin"
    inst.install(archive, dest, "cc", sha256=sha)
    inst.install(archive, dest, "cc", sha256=sha)  # own build: replaced silently
    (dest / inst.VERSION_FILE).unlink()
    inst.install(archive, dest, "cc", sha256=sha, force=True)
    inst.check(dest / "Phi")
