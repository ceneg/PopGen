import numpy as np
import pytest


@pytest.fixture(scope="module")
def rec(load_script):
    return load_script("recombination")


def test_holm(rec):
    assert rec.holm([0.01, None, 0.04, 0.03]) == pytest.approx([0.03, None, 0.06, 0.06])


def test_parse_phi_output(rec):
    out = "Found 12 informative sites.\nPHI (Normal):        1.2e-03\nNSS:  --\n"
    r = rec.parse(out)
    assert r["informative"] == "12"
    assert r["p_phi_normal"] == "1.2e-03"
    assert r["p_nss"] is None


def test_split_segments(rec):
    rng = np.random.default_rng(3)
    h = rng.integers(0, 2, (1000, 12))
    h[::7] = 0  # monomorphic rows, not informative
    n_inf = rec.informative(h)
    assert rec.split_segments(h, n_inf) == [(0, 1000)]
    segs = rec.split_segments(h, 100)
    assert len(segs) == -(-n_inf // 100)  # fewest segments
    assert segs[0][0] == 0 and segs[-1][1] == 1000
    assert all(a[1] == b[0] for a, b in zip(segs, segs[1:]))  # consecutive
    sizes = [rec.informative(h[a:b]) for a, b in segs]
    assert sum(sizes) == n_inf and max(sizes) <= 100 and max(sizes) - min(sizes) <= 1
