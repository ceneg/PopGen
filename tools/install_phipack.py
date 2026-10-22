#!/usr/bin/env python3
"""
Build PhiPack's ``Phi`` program (needed only by scripts/recombination.py) from
its source code and install it next to the running Python interpreter, i.e.
into ``.venv/bin`` when run as ``.venv/bin/python tools/install_phipack.py``.
recombination.py looks for Phi there first, so the virtual environment does
not need to be activated.

The source archive (PhiPack 1.1, Bruen et al. 2006) is downloaded from the
authors' site or, if that fails, from the Galaxy depot mirror used by
bioconda, and checked against the SHA-256 checksum of the bioconda recipe.
With ``--source`` a local copy of the archive is used instead (no download).

Requires a C compiler (``cc``, e.g. gcc or clang).
"""

import argparse
import hashlib
import io
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

log = logging.getLogger("install_phipack")

VERSION = "1.1"
SHA256 = "af43326e021f1f2e590be2cba3caa44a0963f237386e63209ccc26b5bfb02db9"
URLS = (
    "https://www.maths.otago.ac.nz/~dbryant/software/PhiPack.tar.gz",
    "https://depot.galaxyproject.org/software/phipack/phipack_1.1_src_all.tar.gz",
)
# C sources of the Phi program (PhiPack's Makefile, target Phi)
PHI_SOURCES = (
    "normal.c", "stats.c", "maxChi.c", "main.c", "queue.c", "graphCode.c",
    "fasta.c", "phylip.c", "mem.c", "misc.c", "pairScore.c", "seqManip.c",
    "global.c",
)
VERSION_FILE = "PhiPack.version"  # read by scripts/generate_citations.py


def download(urls: tuple[str, ...] = URLS, sha256: str = SHA256) -> bytes:
    """
    First archive from ``urls`` that downloads and matches ``sha256``.

    Raises:
        RuntimeError: If no URL gives a verified archive.
    """
    for url in urls:
        try:
            log.info(f"Downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        except OSError as e:
            log.warning(f"  failed: {e}")
            continue
        try:
            return verify(data, sha256)
        except ValueError as e:
            log.warning(f"  {e}")
    raise RuntimeError(
        "Could not download a verified PhiPack archive. Download it yourself "
        f"(e.g. {URLS[0]}) and run this script with --source <file>."
    )


def verify(data: bytes, sha256: str = SHA256) -> bytes:
    """
    Return ``data`` if its SHA-256 equals ``sha256``.

    Raises:
        ValueError: On a checksum mismatch.
    """
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise ValueError(f"checksum mismatch: {digest} (expected {sha256})")
    return data


def find_source_dir(root: Path) -> Path:
    """
    Directory with Phi's C sources in the unpacked archive (macOS "._*"
    metadata files in the archive are ignored).

    Raises:
        FileNotFoundError: If no directory has all of PHI_SOURCES.
    """
    for main_c in sorted(root.rglob("main.c")):
        if all((main_c.parent / name).is_file() for name in PHI_SOURCES):
            return main_c.parent
    raise FileNotFoundError(f"PhiPack sources not found in {root}")


def build(src_dir: Path, cc: str) -> Path:
    """
    Compile Phi in ``src_dir`` with the flags of PhiPack's Makefile.

    Raises:
        subprocess.CalledProcessError: If compilation fails (stderr is logged).
    """
    cmd = [cc, "-O3", "-o", "Phi", *PHI_SOURCES, "-lm"]
    log.info(f"Compiling: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=src_dir, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log.error(e.stderr)
        raise
    return src_dir / "Phi"


def install(archive: bytes, dest: Path, cc: str, sha256: str = SHA256) -> Path:
    """
    Verify, unpack and compile the archive, and copy Phi to ``dest``.

    Returns:
        Path: The installed Phi program.
    """
    verify(archive, sha256)
    with tempfile.TemporaryDirectory(prefix="phipack.") as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            if hasattr(tarfile, "data_filter"):  # Python >= 3.11.4
                tar.extractall(tmp, filter="data")
            else:
                tar.extractall(tmp)
        phi = build(find_source_dir(Path(tmp)), cc)
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / "Phi"
        shutil.copy2(phi, target)
    target.chmod(0o755)
    (dest / VERSION_FILE).write_text(
        f"PhiPack {VERSION} (built from source, sha256 {sha256[:12]}...)\n"
    )
    return target


def check(phi: Path) -> None:
    """
    Run Phi without arguments; it should print its usage message.

    Raises:
        RuntimeError: If it does not.
    """
    result = subprocess.run([str(phi)], capture_output=True, text=True)
    if "Usage" not in result.stdout + result.stderr:
        raise RuntimeError(f"{phi} does not run as expected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(sys.executable).parent,
        help="Install directory (default: the bin directory of this Python, "
        "e.g. .venv/bin)",
    )
    parser.add_argument(
        "--source", type=Path, help="Local PhiPack.tar.gz instead of downloading"
    )
    parser.add_argument(
        "--cc", default=os.environ.get("CC", "cc"), help="C compiler (default: cc)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if shutil.which(args.cc) is None:
        sys.exit(
            f"Error: C compiler '{args.cc}' not found. Install one (e.g. on "
            "Ubuntu: sudo apt install build-essential) or pass --cc."
        )
    try:
        archive = args.source.read_bytes() if args.source else download()
        phi = install(archive, args.dest, args.cc)
        check(phi)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as e:
        sys.exit(f"Error: {e}")
    log.info(f"Installed {phi}")


if __name__ == "__main__":
    main()
