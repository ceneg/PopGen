"""External programs: locating PhiPack's ``Phi`` and its version."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Written by tools/install_phipack.py next to the Phi program it builds
PHI_VERSION_FILE = "PhiPack.version"


def find_phi() -> str | None:
    """
    Path of PhiPack's ``Phi`` program: the one next to the running Python
    interpreter (installed into the virtual environment by
    ``tools/install_phipack.py``, or a conda environment's), else ``Phi`` on
    the PATH. None if neither exists.
    """
    local = Path(sys.executable).parent / "Phi"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return shutil.which("Phi")


def conda_version(package_name: str) -> str | None:
    """Version of a package in the running conda environment, or None."""
    if not shutil.which("conda"):
        return None

    try:
        # conda list -p PREFIX package_name, for the environment of this Python
        result = subprocess.run(
            ["conda", "list", "-p", sys.prefix, package_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Parse output:
        # # packages in environment at ...
        # #
        # # Name                    Version                   Build  Channel
        # phipack                   1.1                  h7b50bb2_6    bioconda

        for line in result.stdout.splitlines():
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == package_name:
                return parts[1]

    except Exception:
        pass
    return None


def phi_version(phi: str) -> str | None:
    """
    PhiPack version of the ``Phi`` at ``phi``: Phi has no version option, so
    it is read from the file that tools/install_phipack.py writes next to Phi,
    or from conda. None if unknown.
    """
    version_file = Path(phi).with_name(PHI_VERSION_FILE)
    if version_file.is_file():
        return version_file.read_text().strip()
    return conda_version("phipack")
