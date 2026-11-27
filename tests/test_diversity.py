import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from popgen.io import ensure_output_dirs


@pytest.fixture(scope="module")
def div(load_script):
    return load_script("diversity")


def test_failed_sample_exits(tmp_path):
    """B8: a sample without called sites stops the run before any output."""
    from conftest import PROJECT_ROOT, example_vcf_failed_sample, run_script

    vcf = example_vcf_failed_sample(tmp_path / "failed.vcf", "S05")
    meta = PROJECT_ROOT / "examples" / "data" / "metadata.tsv"
    result = run_script("diversity", tmp_path, "--vcf", vcf, "--meta", meta,
                        "--out", tmp_path / "div")
    assert result.returncode != 0
    assert "S05" in result.stdout + result.stderr
    assert not [p for p in (tmp_path / "results").rglob("*") if p.is_file()]


def disagreeing_matrix():
    """6 samples where clustering rows as vectors and UPGMA give different orders."""
    rng = np.random.default_rng(3)
    for _ in range(1000):
        condensed = rng.integers(1, 20, 15).astype(float)
        square = pd.DataFrame(np.zeros((6, 6)))
        square.values[np.triu_indices(6, 1)] = condensed
        square = square + square.T
        upgma = leaves_list(linkage(condensed, "average"))
        rows = leaves_list(linkage(pdist(square.values), "average"))
        if not np.array_equal(upgma, rows):
            return condensed, upgma
    raise AssertionError("no disagreeing matrix found")


def test_heatmap_order_is_upgma_of_distances(div, tmp_path):
    """T-DIV-1 (B18)."""
    condensed, upgma = disagreeing_matrix()
    samples = [f"s{i}" for i in range(6)]
    meta = pd.DataFrame({"habitat": ["a", "b", None, "a", "b", "a"]},
                        index=pd.Index(samples, name="strain"))
    paths = ensure_output_dirs(tmp_path / "d")
    for color_by in (None, "habitat"):
        g = div.plot_heatmap(condensed, samples, meta, "x", paths, "t", color_by)
        assert g.dendrogram_row.reordered_ind == upgma.tolist()
        assert g.dendrogram_col.reordered_ind == upgma.tolist()
    # The colour-bar legend lists the categories, unknown last
    labels = [t.get_text() for t in g.fig.legends[0].get_texts()]
    assert labels == ["a", "b", "unknown"]


def test_one_sample_exits_before_output(tmp_path):
    from conftest import run_script

    vcf = tmp_path / "one.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\n"
        "c1\t5\t.\tA\tC\t.\tPASS\t.\tGT\t1\n"
    )
    meta = tmp_path / "m.tsv"
    meta.write_text("strain\tg\na\tx\n")
    result = run_script("diversity", tmp_path, "--vcf", vcf, "--meta", meta,
                        "--out", tmp_path / "div")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "at least 2 samples" in result.stdout + result.stderr
    assert not [p for p in (tmp_path / "results").rglob("*") if p.is_file()]
