"""
Golden outputs: the example pipeline must reproduce tests/golden/ byte for byte.

Regenerate with ``POPGEN_UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py``
(see tests/golden/README.md for when that is allowed).
"""

import filecmp
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from conftest import PROJECT_ROOT
from popgen.external import find_phi

DATA = PROJECT_ROOT / "examples" / "data"
VCF = DATA / "example.vcf"
META = DATA / "metadata.tsv"
GOLDEN = PROJECT_ROOT / "tests" / "golden"
UPDATE = os.environ.get("POPGEN_UPDATE_GOLDEN") == "1"

TABLES = [
    "ex.pca.tsv",
    "ex.dapc.tsv",
    "ex.upgma.nwk",
    "div.nd.tsv",
    "div.ss.tsv",
    "net.genotype_summary.txt",
    "ld.ld_decay.tsv",
    "sel.tsv",
    "sel.summary.tsv",
]
FIGURES = [
    "ex.pca",
    "ex.pca.lineage.annotated",
    "ex.dapc.clusters",
    "ex.upgma",
    "div.nd",
    "div.ss.lineage",
    "net.lineage.msn",
    "ld.ld_decay",
    "sel.dN_dS.upgma",
]
REC_TABLE = "rec.phi.contigs.tsv"
# Permutation p-values are random; only these columns are deterministic
REC_COLUMNS = ["contig", "snps", "informative", "phi_obs"]


def run(script, tmp, *args):
    """Run scripts/<script>.py with logs in tmp."""
    env = dict(os.environ, POPGEN_LOG_DIR=str(tmp / "logs"))
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / f"{script}.py"), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Run the example pipeline once, as tests/test_pipeline.py does."""
    tmp = tmp_path_factory.mktemp("golden")
    run("structure", tmp, "--vcf", VCF, "--meta", META, "--out", tmp / "ex")
    run("diversity", tmp, "--vcf", VCF, "--meta", META, "--out", tmp / "div")
    run("network", tmp, "--vcf", VCF, "--meta", META, "--contract", "5",
        "--diameter", "0.05", "--out", tmp / "net")
    run("ld_decay", tmp, "--vcf", VCF, "--out", tmp / "ld")
    run("selection", tmp, "--vcf", VCF, "--gff", DATA / "example.gff",
        "--fasta", DATA / "example.fna", "--nwk", tmp / "results" / "ex.upgma.nwk",
        "--out", tmp / "sel")
    return tmp / "results"


@pytest.mark.parametrize("name", TABLES)
def test_table_matches_golden(results, name):
    if UPDATE:
        GOLDEN.mkdir(exist_ok=True)
        shutil.copyfile(results / name, GOLDEN / name)
    assert filecmp.cmp(results / name, GOLDEN / name, shallow=False), name


@pytest.mark.parametrize("stem", FIGURES)
def test_figure_written(results, stem):
    for kind in ("png", "svg"):
        path = results / kind / f"{stem}.{kind}"
        assert path.is_file() and path.stat().st_size > 0, path


@pytest.mark.skipif(find_phi() is None, reason="PhiPack (Phi) not installed")
def test_recombination_matches_golden(tmp_path):
    run("recombination", tmp_path, "--vcf", VCF, "--out", tmp_path / "rec",
        "--jobs", "2")
    new = tmp_path / "results" / REC_TABLE
    if UPDATE:
        shutil.copyfile(new, GOLDEN / REC_TABLE)
    got = pd.read_csv(new, sep="\t", dtype=str)[REC_COLUMNS]
    want = pd.read_csv(GOLDEN / REC_TABLE, sep="\t", dtype=str)[REC_COLUMNS]
    pd.testing.assert_frame_equal(got, want)


def test_example_data_regenerates(tmp_path):
    """examples/make_example_data.py reproduces examples/data exactly (seeded)."""
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "examples" / "make_example_data.py"),
         "--outdir", str(tmp_path)],
        check=True, capture_output=True,
    )
    for name in ("example.fna", "example.gff", "example.vcf", "metadata.tsv"):
        assert filecmp.cmp(tmp_path / name, DATA / name, shallow=False), name
