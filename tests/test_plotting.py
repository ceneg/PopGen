import numpy as np
import pandas as pd
import pytest

from popgen.io import ensure_output_dirs
from popgen.plotting import UNKNOWN, UNKNOWN_COLOR, categories, category_colors


def test_category_colors_sorted_with_unknown_last():
    """T-PLOT-1 (R5)."""
    colors = category_colors(["b", None, "a"])
    assert list(colors) == ["a", "b", UNKNOWN]
    assert colors[UNKNOWN] == UNKNOWN_COLOR
    assert UNKNOWN not in category_colors(["b", "a"])


def test_category_colors_distinct_above_20():
    colors = category_colors([f"c{i}" for i in range(21)])
    assert len(set(colors.values())) == 21


def test_categories_labels_missing_as_unknown():
    s = pd.Series(["x", np.nan, None], index=["a", "b", "c"])
    assert categories(s).tolist() == ["x", UNKNOWN, UNKNOWN]
    assert list(categories(s).index) == ["a", "b", "c"]


@pytest.fixture
def capture_figures(monkeypatch, load_script):
    """Replace save_figure in the analyses with a recorder of the figures."""
    figs = []
    for name in ("structure", "network", "diversity"):
        monkeypatch.setattr(load_script(name), "save_figure",
                            lambda fig, paths, stem, **kw: figs.append(fig))
    return figs


def legend_colors(fig):
    """{label: RGBA} of every legend in the figure (axes and figure legends)."""
    out = {}
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes]
    for leg in filter(None, legends):
        for handle, text in zip(leg.legend_handles, leg.get_texts()):
            out[text.get_text()] = tuple(handle.get_facecolor())
    return out


def test_same_colour_in_every_script(load_script, capture_figures, tmp_path):
    """T-PLOT-1 (R5): 'soil' has one colour in structure, network and diversity."""
    import networkx as nx

    samples = ["s1", "s2", "s3", "s4"]
    meta = pd.DataFrame({"habitat": ["water", "soil", None, "indoor"]},
                        index=pd.Index(samples, name="strain"))
    paths = ensure_output_dirs(tmp_path / "p")

    st = load_script("structure")
    df = pd.DataFrame({"PC1": [0.0, 1, 2, 3], "PC2": [1.0, 0, 1, 0]}, index=samples)
    st.plot_by_metadata(df, "PC1", "PC2", meta, paths, "pca", "PCA")

    net = load_script("network")
    G = nx.Graph()
    G.add_edge("G1", "G2", weight=1)
    net.draw_msn(G, {"s1": "G1", "s2": "G1", "s3": "G2", "s4": "G2"}, meta,
                 "habitat", paths)

    div = load_script("diversity")
    div.plot_heatmap(np.arange(1.0, 7.0), samples, meta, "x", paths, "nd", "habitat")

    expected = category_colors(meta["habitat"])
    assert len(capture_figures) == 4  # structure x2, network, diversity
    for fig in capture_figures:
        got = legend_colors(fig)
        assert list(got) == ["indoor", "soil", "water", UNKNOWN]
        for label, rgb in expected.items():
            assert got[label][:3] == pytest.approx(tuple(rgb))
