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


@pytest.mark.xfail(strict=True, reason="B5")
def test_half_decay_none_when_peak_below_half(ld):
    # Bin centres; the curve never exceeds 0.45, so it cannot fall to it.
    x = np.arange(50, 1000, 100)
    y = np.array([0.2, 0.3, 0.35, 0.3, 0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
    assert ld.half_decay_distance(x, y, 0.45) is None  # v1.01: 450


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
