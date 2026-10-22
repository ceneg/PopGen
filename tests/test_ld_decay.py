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
