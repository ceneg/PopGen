import numpy as np
import pytest
from Bio import Phylo


@pytest.fixture(scope="module")
def st(load_script):
    return load_script("structure")


def test_ellipse_scale_is_95_percent(st):
    assert st.ELLIPSE_N_STD == pytest.approx(2.4477, abs=1e-4)


def test_impute_missing_uses_column_mean(st):
    x = st.impute_missing(np.array([[0, -1], [1, 1], [1, 0]]))
    np.testing.assert_allclose(x, [[0, 0.5], [1, 1], [1, 0]])


def test_newick_patristic_distance_equals_upgma(st, tmp_path):
    g = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1]])
    dist = st.compute_distance_matrix(g)
    paths = {k: str(tmp_path / "t") for k in ("png", "svg", "tsv")}
    st.plot_tree(dist, ["a", "b", "c"], paths)
    tree = Phylo.read(str(tmp_path / "t.upgma.nwk"), "newick")
    assert tree.distance("a", "b") == pytest.approx(dist[0, 1])


def test_dapc_n_pcs_skips_low_variance_pcs(st):
    evr = np.array([0.55, 0.31, 0.08, 0.03, 0.01, 0.0])
    assert st.dapc_n_pcs(evr, 92) == 3
    assert st.dapc_n_pcs(np.array([0.99, 0.01]), 92) == 2  # at least 2


def test_choose_dapc_clusters_avoids_tiny_clusters(st):
    rng = np.random.default_rng(1)
    centres = np.array([[0, 0], [10, 0], [0, 10]])
    coords = np.vstack([c + rng.normal(0, 0.5, (10, 2)) for c in centres])
    labels, k = st.choose_dapc_clusters(coords)
    assert k >= 3
    assert np.bincount(labels).min() >= st.DAPC_MIN_CLUSTER
    # a lone outlier: no k qualifies, k = 3 is used
    labels, k = st.choose_dapc_clusters(np.vstack([coords[:20], [[30, 30]]]))
    assert k == 3
