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


GOLDEN = __import__("conftest").PROJECT_ROOT / "tests" / "golden"


def near_identical_genotypes():
    """12 samples x 80 sites: 3 haplotypes with 0-4 private flips, 10% missing."""
    rng = np.random.default_rng(11)
    bases = rng.integers(0, 2, (3, 80))
    g = bases[np.arange(12) % 3].copy()
    for i in range(12):
        flips = rng.choice(80, rng.integers(0, 5), replace=False)
        g[i, flips] = 1 - g[i, flips]
    g[rng.random(g.shape) < 0.1] = -1
    return [f"s{i}" for i in range(12)], g


def test_collapse_matches_v101_record(net):
    """Collapse and representative distances as recorded from v1.01."""
    import json
    import os

    names, g = near_identical_genotypes()
    got = {}
    for threshold in (0, 2, 5):
        cluster_map, clusters = net.collapse_sequences(names, g, threshold)
        gnames, dist = net.compute_distance_matrix(clusters)
        got[str(threshold)] = {"cluster_map": cluster_map, "groups": gnames,
                               "dist": dist.tolist()}
    record = GOLDEN / "net_collapse.json"
    if os.environ.get("POPGEN_UPDATE_GOLDEN") == "1":
        record.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    assert got == json.loads(record.read_text())
    assert len(got["0"]["groups"]) > len(got["5"]["groups"])  # threshold matters


def test_one_genotype_without_metadata(tmp_path):
    from conftest import run_script

    header = (
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\tc\n"
    )
    vcf = tmp_path / "same.vcf"
    vcf.write_text(header + "c1\t5\t.\tA\tC\t.\tPASS\t.\tGT\t1\t1\t1\n"
                   + "c1\t9\t.\tG\tT\t.\tPASS\t.\tGT\t0\t0\t0\n")
    result = run_script("network", tmp_path, "--vcf", vcf, "--out", tmp_path / "net")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "results" / "png" / "net.group.msn.png").stat().st_size > 0
    summary = (tmp_path / "results" / "net.genotype_summary.txt").read_text()
    assert summary == "#G1\na\nb\nc\n\n"
