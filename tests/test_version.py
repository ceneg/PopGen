"""T-VERSION: the version number is the same everywhere it is written."""

import re

import popgen
from conftest import PROJECT_ROOT


def test_version_matches_citation_and_readme():
    cff = (PROJECT_ROOT / "CITATION.cff").read_text()
    readme = (PROJECT_ROOT / "README.md").read_text()
    cff_version = re.search(r'^version:\s*"?([^"\s]+)"?', cff, re.M).group(1)
    badge_version = re.search(r"badge/version-([0-9.]+)-", readme).group(1)
    assert popgen.__version__ == cff_version == badge_version
