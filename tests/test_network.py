import numpy as np
import pytest


@pytest.fixture(scope="module")
def net(load_script):
    return load_script("network")


def test_collapse_ignores_missing(net):
    g = np.array([[0, 1, 0], [0, -1, 0], [1, 0, 1]])
    cluster_map, clusters = net.collapse_sequences(["a", "b", "c"], g, 0)
    assert cluster_map == {"a": "G1", "b": "G1", "c": "G2"}
    names, dist = net.compute_distance_matrix(clusters)
    assert names == ["G1", "G2"]
    assert dist[0, 1] == 3


def test_layout_draws_short_edges_shorter(net):
    import networkx as nx

    G = nx.Graph()
    G.add_edge("a", "b", weight=100)
    G.add_edge("b", "c", weight=1)
    pos = net.layout_tree(G)
    assert np.linalg.norm(pos["b"] - pos["c"]) < np.linalg.norm(pos["a"] - pos["b"])


def test_msn_is_minimum_spanning_tree(net):
    from scipy.sparse.csgraph import minimum_spanning_tree

    rng = np.random.default_rng(0)
    g = rng.integers(0, 2, (9, 60))
    _, clusters = net.collapse_sequences([str(i) for i in range(9)], g, 0)
    names, dist = net.compute_distance_matrix(clusters)
    tree = net.build_msn(names, dist)
    assert tree.number_of_edges() == len(names) - 1
    total = sum(d["weight"] for _, _, d in tree.edges(data=True))
    assert total == minimum_spanning_tree(dist).sum()
