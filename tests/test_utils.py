import numpy as np
import pytest
from scipy.spatial.distance import pdist

from src.utils import (
    alt_allele_counts,
    ensure_output_dirs,
    pairwise_differences,
    read_vcf,
    setup_logging,
)

VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
)


def test_pairwise_differences_no_missing():
    g = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 0]])
    diffs, compared = pairwise_differences(g)
    np.testing.assert_array_equal(diffs, pdist(g, "hamming") * 4)
    np.testing.assert_array_equal(compared, [4, 4, 4])


def test_pairwise_differences_missing_ignored():
    g = np.array([[0, 1, -1, 0], [0, 0, 1, -1], [1, 1, 1, 0]])
    diffs, compared = pairwise_differences(g)
    # pairs (0,1), (0,2), (1,2)
    np.testing.assert_array_equal(diffs, [1, 1, 2])
    np.testing.assert_array_equal(compared, [2, 3, 3])


def test_pairwise_differences_chunked_matches_unchunked():
    rng = np.random.default_rng(3)
    g = rng.integers(-1, 3, (6, 101))
    full = pairwise_differences(g, chunk=1000)
    for chunk in (1, 7, 50):
        for a, b in zip(pairwise_differences(g, chunk=chunk), full):
            np.testing.assert_array_equal(a, b)


def test_setup_logging_writes_run_details(tmp_path):
    import argparse
    import logging

    log_file = setup_logging("demo", argparse.Namespace(vcf="x.vcf"), tmp_path)
    logging.getLogger("demo").info("hello")
    for h in logging.getLogger().handlers:
        h.flush()
    text = log_file.read_text()
    assert "Parameters: {'vcf': 'x.vcf'}" in text
    assert "numpy=" in text and "hello" in text
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()


def test_read_vcf_rejects_multiallelic(tmp_path):
    vcf = tmp_path / "multi.vcf"
    vcf.write_text(VCF_HEADER + "c1\t5\t.\tA\tC,G\t.\tPASS\t.\tGT\t1\t2\n")
    with pytest.raises(SystemExit):
        read_vcf(str(vcf), fields=["samples", "calldata/GT"])


def test_read_vcf_biallelic(tmp_path):
    vcf = tmp_path / "ok.vcf"
    vcf.write_text(VCF_HEADER + "c1\t5\t.\tA\tC\t.\tPASS\t.\tGT\t0\t1\n")
    samples, gt, _ = read_vcf(str(vcf), fields=["samples", "calldata/GT"])
    assert list(samples) == ["A", "B"]
    assert gt.shape[:2] == (1, 2)


def test_ensure_output_dirs(tmp_path):
    paths = ensure_output_dirs(str(tmp_path / "run"))
    assert (tmp_path / "results" / "png").is_dir()
    assert (tmp_path / "results" / "svg").is_dir()
    assert paths["tsv"] == str(tmp_path / "results" / "run")


def test_alt_allele_counts_haploid_parsed_as_diploid():
    # scikit-allel reads haploid "1" / "0" / "." as (1,-1), (0,-1), (-1,-1)
    g = np.array([[[1, -1], [0, -1], [-1, -1]]])
    np.testing.assert_array_equal(alt_allele_counts(g), [[1, 0, -1]])


def test_alt_allele_counts_diploid():
    g = np.array([[[0, 1], [1, 1], [0, -1]]])
    np.testing.assert_array_equal(alt_allele_counts(g), [[1, 2, -1]])
