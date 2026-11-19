import numpy as np
import pytest
from scipy.spatial.distance import pdist

from popgen.utils import alt_allele_counts, load_metadata, pairwise_differences


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


# --- Metadata (B1). load_metadata is moved to popgen.io in Phase 3. ---

def write_meta(tmp_path, text):
    path = tmp_path / "meta.tsv"
    path.write_text(text)
    return str(path)


def test_load_metadata_follows_vcf_order(tmp_path):
    meta = write_meta(tmp_path, "strain\tg\nB\tx\nZ\tz\nA\ty\n")
    out = load_metadata(meta, ["A", "B"])
    assert out.index.name == "strain"
    assert list(out.index) == ["A", "B"]  # VCF order; surplus Z dropped
    assert list(out["g"]) == ["y", "x"]


@pytest.mark.xfail(strict=True, reason="B1")
def test_load_metadata_numeric_names_stay_strings(tmp_path):
    meta = write_meta(tmp_path, "strain\tyear\n101\t2019\n102\t2020\n")
    out = load_metadata(meta, ["101", "102"])
    assert list(out.index) == ["101", "102"]
    assert list(out["year"]) == ["2019", "2020"]


@pytest.mark.xfail(strict=True, reason="B1")
def test_load_metadata_no_overlap_exits(tmp_path):
    meta = write_meta(tmp_path, "strain\tyear\n101\t2019\n102\t2020\n")
    with pytest.raises(SystemExit):
        load_metadata(meta, ["X", "Y"])


@pytest.mark.xfail(strict=True, reason="B1")
def test_load_metadata_duplicate_strain_exits(tmp_path):
    meta = write_meta(tmp_path, "strain\tg\nA\tx\nA\ty\nB\tz\n")
    with pytest.raises(SystemExit):
        load_metadata(meta, ["A", "B"])


@pytest.mark.xfail(strict=True, reason="B1")
def test_load_metadata_strain_only(tmp_path):
    meta = write_meta(tmp_path, "strain\nA\nB\n")
    out = load_metadata(meta, ["A", "B"])
    assert out.shape == (2, 0)
    assert list(out.index) == ["A", "B"]


@pytest.mark.xfail(strict=True, reason="B1")
def test_load_metadata_keeps_samples_without_metadata(tmp_path, caplog):
    meta = write_meta(tmp_path, "strain\tg\nA\tx\nB\ty\n")
    with caplog.at_level("WARNING"):
        out = load_metadata(meta, ["A", "B", "C"])
    assert list(out.index) == ["A", "B", "C"]
    assert out.loc["C"].isna().all()
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
