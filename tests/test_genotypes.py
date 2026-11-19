import numpy as np
import pytest
from scipy.spatial.distance import pdist

from popgen.genotypes import (
    alt_allele_counts,
    call_states,
    contig_rows,
    haploid_alleles,
    pairwise_differences,
    ploidy_kind,
    report_call_rates,
    require_shared_sites,
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


def test_alt_allele_counts_haploid_parsed_as_diploid():
    # scikit-allel reads haploid "1" / "0" / "." as (1,-1), (0,-1), (-1,-1)
    g = np.array([[[1, -1], [0, -1], [-1, -1]]])
    np.testing.assert_array_equal(alt_allele_counts(g), [[1, 0, -1]])


def test_alt_allele_counts_diploid():
    g = np.array([[[0, 1], [1, 1], [0, -1]]])
    np.testing.assert_array_equal(alt_allele_counts(g), [[1, 2, -1]])


def test_mixed_ploidy_mostly_haploid_exits(caplog):
    """T-GT-1 (B2): two haploid samples, one diploid."""
    g = np.array([[[0, -1], [1, -1], [0, 1]], [[1, -1], [0, -1], [0, 0]]])
    with pytest.raises(SystemExit):
        alt_allele_counts(g)
    assert "4 haploid calls" in caplog.text and "2 diploid calls" in caplog.text


def test_mixed_ploidy_mostly_diploid_warns(caplog):
    """T-GT-2 (B2): one half call among many full calls keeps today's result."""
    g = np.zeros((10, 3, 2), dtype=int)
    g[0, 0] = [1, 1]
    g[3, 2] = [1, -1]
    expected = np.zeros((10, 3), dtype=int)
    expected[0, 0] = 2
    expected[3, 2] = -1
    with caplog.at_level("WARNING"):
        np.testing.assert_array_equal(alt_allele_counts(g), expected)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "1 haploid calls" in warnings[0]


def test_call_states():
    half, full = call_states(np.array([[[0, -1], [0, 1], [-1, -1], [-1, 1]]]))
    np.testing.assert_array_equal(half, [[True, False, False, False]])
    np.testing.assert_array_equal(full, [[False, True, False, False]])


@pytest.mark.parametrize("g, kind", [
    ([[[0, -1], [1, -1]]], "haploid"),
    ([[[0, 1], [1, 1]]], "diploid"),
    ([[[0, -1], [1, 1]]], "mixed"),
    (np.zeros((0, 3, 2), dtype=int), "haploid"),
    ([[0, 1], [1, -1]], "haploid"),  # 2-D
    ([[[0], [1]]], "haploid"),  # ploidy 1
])
def test_ploidy_kind(g, kind):
    """T-GT-3."""
    assert ploidy_kind(np.array(g)) == kind


def test_haploid_alleles(caplog):
    """T-GT-4."""
    with pytest.raises(SystemExit):
        haploid_alleles(np.array([[[0, 1]]]), "error")
    with caplog.at_level("WARNING"):
        np.testing.assert_array_equal(haploid_alleles(np.array([[[0, 1]]]), "first"),
                                      [[0]])
    assert "1 heterozygous" in caplog.text
    g = np.array([[[1, -1], [-1, -1], [1, 1]]])
    np.testing.assert_array_equal(haploid_alleles(g, "error"), [[1, -1, 1]])


def test_report_call_rates(caplog):
    counts = np.array([[0, 1, 1, 0], [-1, -1, -1, 0], [1, 1, 0, 0]])
    with caplog.at_level("WARNING"):
        rate = report_call_rates(counts, ["a", "b", "c"])
    np.testing.assert_allclose(rate, [1, 0.25, 1])
    assert "b (25.0%)" in caplog.text and "a (" not in caplog.text


def test_require_shared_sites_exits(caplog):
    """T-GT-5 (B8)."""
    counts = np.array([[0, 1, 1, 0], [-1, -1, -1, -1], [1, 1, 0, 0]])
    _, compared = pairwise_differences(counts)
    with pytest.raises(SystemExit):
        require_shared_sites(compared, ["s1", "s2", "s3"])
    assert "s2" in caplog.text
    require_shared_sites(pairwise_differences(counts[[0, 2]])[1], ["s1", "s3"])


def test_contig_rows():
    """T-GT-6 (B19)."""
    rows = contig_rows(["a", "b", "a"], [5, 1, 9], "error")
    assert list(rows) == ["a", "b"]
    np.testing.assert_array_equal(rows["a"], [0, 2])
    np.testing.assert_array_equal(rows["b"], [1])
    with pytest.raises(SystemExit):
        contig_rows(["a", "a", "a"], [9, 1, 5], "error")
    np.testing.assert_array_equal(contig_rows(["a"] * 3, [9, 1, 5], "sort")["a"],
                                  [1, 2, 0])
    assert contig_rows(np.array([], dtype=object), np.array([]), "error") == {}


def test_contig_rows_first_appearance_order():
    chrom = np.array(["z", "b", "z", "a", "b"], dtype=object)
    rows = contig_rows(chrom, [1, 1, 2, 1, 2], "error")
    assert list(rows) == ["z", "b", "a"]
    np.testing.assert_array_equal(rows["b"], [1, 4])
