"""
End-to-end runs of every script on the synthetic data in examples/data.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import PROJECT_ROOT
from src.utils import find_phi

DATA = PROJECT_ROOT / "examples" / "data"
VCF = DATA / "example.vcf"
META = DATA / "metadata.tsv"


def run(script, tmp, *args):
    """Run scripts/<script>.py with logs in tmp; return its stdout."""
    env = dict(os.environ, POPGEN_LOG_DIR=str(tmp / "logs"))
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / f"{script}.py"), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def outdir(tmp_path_factory):
    """Run structure.py once (its tree is an input of selection.py)."""
    tmp = tmp_path_factory.mktemp("pipeline")
    stdout = run("structure", tmp, "--vcf", VCF, "--meta", META, "--out", tmp / "ex")
    return tmp, stdout


def test_structure_recovers_simulated_lineages(outdir):
    tmp, stdout = outdir
    res = tmp / "results"
    assert "k = 3 clusters" in stdout and "sizes [8, 8, 8]" in stdout
    assert len(pd.read_csv(res / "ex.pca.tsv", sep="\t")) == 24
    assert (res / "ex.upgma.nwk").read_text().endswith(";")
    assert (res / "png" / "ex.dapc.lineage.annotated.png").exists()


def test_diversity(outdir):
    tmp, _ = outdir
    run("diversity", tmp, "--vcf", VCF, "--meta", META, "--out", tmp / "div")
    nd = pd.read_csv(tmp / "results" / "div.nd.tsv", sep="\t", index_col=0)
    assert nd.shape == (24, 24)
    assert np.allclose(np.diag(nd), 0) and np.allclose(nd, nd.T)
    assert (tmp / "results" / "svg" / "div.ss.lineage.svg").exists()


def test_network(outdir):
    tmp, _ = outdir
    run("network", tmp, "--vcf", VCF, "--meta", META, "--contract", "5",
        "--diameter", "0.05", "--out", tmp / "net")
    summary = (tmp / "results" / "net.genotype_summary.txt").read_text()
    assert summary.startswith("#G1")
    assert (tmp / "results" / "png" / "net.lineage.msn.png").exists()


def test_ld_decay(outdir):
    tmp, _ = outdir
    stdout = run("ld_decay", tmp, "--vcf", VCF, "--out", tmp / "ld")
    binned = pd.read_csv(tmp / "results" / "ld.ld_decay.tsv", sep="\t")
    assert len(binned) > 50 and binned["r2"].between(0, 1).all()
    assert "Half-decay distance" in stdout  # simulated recombination


def test_selection_neutral_mutations(outdir):
    tmp, _ = outdir
    run("selection", tmp, "--vcf", VCF, "--gff", DATA / "example.gff",
        "--fasta", DATA / "example.fna", "--nwk", tmp / "results" / "ex.upgma.nwk",
        "--out", tmp / "sel")
    summary = pd.read_csv(tmp / "results" / "sel.summary.tsv", sep="\t", index_col=0)
    assert len(summary) == 24
    # Simulated SNPs are random (neutral) changes: dN/dS should be near 1
    assert 0.6 < summary["dN/dS"].median() < 1.6
    assert (tmp / "results" / "png" / "sel.dN_dS.upgma.png").exists()


@pytest.mark.skipif(find_phi() is None, reason="PhiPack (Phi) not installed")
def test_recombination(outdir):
    tmp, _ = outdir
    run("recombination", tmp, "--vcf", VCF, "--out", tmp / "rec", "--jobs", "2")
    tsv = pd.read_csv(tmp / "results" / "rec.phi.contigs.tsv", sep="\t")
    assert list(tsv["contig"]) == ["ctg1", "ctg2"]
    assert (tsv["p_phi_holm"] < 0.05).all()  # simulated recombination
