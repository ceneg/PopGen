import numpy as np
import pytest


@pytest.fixture(scope="module")
def ld(load_script):
    return load_script("ld_decay")


def test_pairwise_r2_matches_corrcoef(ld):
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, 30)
    nb = rng.integers(0, 2, (5, 30))
    expected = [np.corrcoef(t, row)[0, 1] ** 2 for row in nb]
    np.testing.assert_allclose(ld.pairwise_r2(t, nb), expected, atol=1e-12)


def test_pairwise_r2_ignores_missing(ld):
    t = np.array([0, 1, 0, 1, -1])
    nb = np.array([[0, 1, 0, 1, 0], [0, 0, 0, 0, 1]])
    r2 = ld.pairwise_r2(t, nb)
    assert r2[0] == pytest.approx(1.0)
    assert np.isnan(r2[1])  # monomorphic among shared samples


def test_half_decay_distance_first_crossing(ld):
    x = np.array([0, 100, 200, 300, 400])
    y = np.array([0.8, 0.6, 0.2, 0.5, 0.1])
    assert ld.half_decay_distance(x, y, 0.4) == pytest.approx(150.0)
    assert ld.half_decay_distance(x, y, 0.01) is None


def test_windowed_ld_bins_match_direct_means(ld):
    rng = np.random.default_rng(5)
    gn = rng.integers(0, 2, (40, 20))
    pos = np.sort(rng.choice(np.arange(1, 2000), 40, replace=False))
    s, n = ld.calculate_windowed_ld(gn, pos, 500, bin_size=100)
    dist, r2 = [], []
    for i in range(40):
        for j in range(i + 1, 40):
            if pos[j] - pos[i] <= 500:
                r = ld.pairwise_r2(gn[i], gn[j][None, :])[0]
                if not np.isnan(r):
                    dist.append(pos[j] - pos[i])
                    r2.append(r)
    dist, r2 = np.array(dist), np.array(r2)
    assert n.sum() == len(dist)
    for k in range(len(n)):
        sel = (dist > k * 100) & (dist <= (k + 1) * 100)
        assert n[k] == sel.sum()
        if sel.any():
            assert s[k] / n[k] == pytest.approx(r2[sel].mean())


def test_half_decay_none_when_peak_below_half(ld):
    """T-LD-1 (B5)."""
    # Bin centres; the curve never exceeds 0.45, so it cannot fall to it.
    x = np.arange(50, 1000, 100)
    y = np.array([0.2, 0.3, 0.35, 0.3, 0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
    assert ld.half_decay_distance(x, y, 0.45) is None  # v1.01: 450
    assert ld.half_decay_distance([], [], 0.1) is None


def test_filter_variants_haploid_and_diploid(ld):
    """T-LD-3 (B2): min 2 carriers, at most 10% missing (10 samples)."""
    hap = np.full((3, 10), 0)
    hap[0, :2] = 1  # two carriers: kept
    hap[1, :1] = 1  # one carrier: dropped
    hap[2, :3] = 1
    hap[2, 9] = -1  # 1 of 10 missing: kept
    g = np.stack([hap, np.full_like(hap, -1)], axis=2)  # "x/."
    keep, gn = ld.filter_variants_min_samples(g, 2, 0.1)
    np.testing.assert_array_equal(keep, [True, False, True])
    np.testing.assert_array_equal(gn, hap)

    dip = np.zeros((3, 10, 2), dtype=int)
    dip[0, 0] = [1, 1]  # one carrier (homozygous): dropped
    dip[1, :2, 1] = 1  # two heterozygous carriers: kept
    dip[2, :2, 1] = 1
    dip[2, 8:, 0] = -1  # 2 of 10 missing: dropped
    keep, gn = ld.filter_variants_min_samples(dip, 2, 0.1)
    np.testing.assert_array_equal(keep, [False, True, False])
    assert gn[0, 0] == 2 and gn[2, 9] == -1


def test_unsorted_vcf_exits(tmp_path):
    """T-LD-4 (B19)."""
    from conftest import run_script

    header = (
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\tc\n"
    )
    body = "".join(f"c1\t{p}\t.\tA\tC\t.\tPASS\t.\tGT\t1\t1\t0\n"
                   for p in (500, 100, 300))
    vcf = tmp_path / "unsorted.vcf"
    vcf.write_text(header + body)
    result = run_script("ld_decay", tmp_path, "--vcf", vcf, "--out", tmp_path / "ld")
    assert result.returncode != 0
    assert "not sorted" in result.stdout + result.stderr


def logged_half_decay(ld, r2_sum, n_pairs, tmp_path, caplog):
    import re

    from popgen.io import ensure_output_dirs

    caplog.clear()
    with caplog.at_level("INFO", logger="ld_decay"):
        ld.plot_ld_decay(r2_sum, n_pairs, ensure_output_dirs(tmp_path / "ld"))
    found = re.findall(r"Half-decay distance: (\d+) bp", caplog.text)
    assert len(found) == 1, caplog.text
    return float(found[0])


def test_sparse_outlier_bin_does_not_move_half_decay(ld, tmp_path, caplog):
    """T-LD-2 (B5): one bin with one pair at r2 = 1 far out."""
    import pandas as pd

    centres = np.arange(100) * 100 + 50
    n_pairs = np.full(100, 500)
    # Bin means need some noise: on an exact curve LOWESS's robustness weights
    # (residual / median residual) degenerate and keep the outlier
    noise = np.random.default_rng(0).normal(0, 0.02, 100)
    r2_sum = (0.05 + 0.6 * np.exp(-centres / 1500) + noise) * n_pairs
    clean = logged_half_decay(ld, r2_sum, n_pairs, tmp_path, caplog)

    r2_sum[90], n_pairs[90] = 1.0, 1
    with_outlier = logged_half_decay(ld, r2_sum, n_pairs, tmp_path, caplog)
    assert abs(with_outlier - clean) <= 100
    table = pd.read_csv(tmp_path / "results" / "ld.ld_decay.tsv", sep="\t")
    assert list(table.columns) == ["dist", "r2", "n_pairs"]
    assert table["n_pairs"].sum() == n_pairs.sum()


def test_too_few_bins_skips_smoothing(ld, tmp_path, caplog):
    from popgen.io import ensure_output_dirs

    with caplog.at_level("INFO", logger="ld_decay"):
        ld.plot_ld_decay(np.array([0.5, 0.2]), np.array([1, 1]),
                         ensure_output_dirs(tmp_path / "ld"))
    assert "Too few bins to smooth" in caplog.text
    assert (tmp_path / "results" / "ld.ld_decay.tsv").exists()


def test_windowed_ld_int32_positions_do_not_overflow(ld):
    """VCF POS is int32: pos + window above 2**31 - 1 must not wrap."""
    gn = np.array([[0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0]])
    pos = np.array([100, 200, 300], dtype=np.int32)
    s, n = ld.calculate_windowed_ld(gn, pos, 2**31 - 1, bin_size=2**30)
    assert n.sum() == 3
