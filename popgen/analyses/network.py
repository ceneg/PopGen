"""
Genotype network: collapse near-identical samples into genotypes, connect the
genotypes by a minimum spanning tree of their pairwise differences, and draw
it with a pie chart of metadata categories at every node.
"""

import argparse
import logging
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.tree import minimum_spanning_tree as nx_mst
from scipy.spatial.distance import squareform

from popgen.cli import add_out_argument, non_negative_int, positive_float
from popgen.genotypes import (
    alt_allele_counts,
    distance_matrices,
    report_call_rates,
    require_shared_sites,
)
from popgen.io import (
    OutputPaths,
    atomic_write,
    ensure_output_dirs,
    load_metadata,
    read_vcf,
    safe_name,
    safe_names,
)
from popgen.plotting import (
    add_category_legend,
    categories,
    category_colors,
    save_figure,
)
from popgen.runlog import setup_logging

log = logging.getLogger("network")


def collapse_sequences(
    names, diff_square: np.ndarray, threshold: int
) -> tuple[dict[str, str], list[int]]:
    """
    Group samples whose genotypes differ at no more than ``threshold`` sites.

    Greedy, in input order: each sample joins the first group whose
    representative (its first member) is within the threshold.

    Args:
        names: Sample names.
        diff_square: Square matrix of differing sites between samples (sites
            called in both), from ``distance_matrices``.
        threshold: Largest number of differences within a group.

    Returns:
        tuple: ({sample: "G<n>"}, sample indices of the group representatives).
        The representatives' distances are ``diff_square[np.ix_(reps, reps)]``.
    """
    log.info(f"Clustering sequences with contract threshold {threshold}...")
    reps = []
    cluster_map = {}

    for i, name in enumerate(names):
        for idx, rep in enumerate(reps):
            if diff_square[i, rep] <= threshold:
                cluster_map[name] = f"G{idx + 1}"
                break
        else:
            reps.append(i)
            cluster_map[name] = f"G{len(reps)}"

    log.info(f"Identified {len(reps)} unique genotypes (clusters).")
    return cluster_map, reps


def build_msn(gnames, dist_matrix) -> nx.Graph:
    """
    Minimum spanning tree (Kruskal) of the genotype distance graph. Where several
    trees have the same length, one of them is returned.
    """
    log.info("Building minimum spanning tree...")
    G = nx.Graph()
    G.add_nodes_from(gnames)

    for i in range(len(gnames)):
        for j in range(i + 1, len(gnames)):
            G.add_edge(gnames[i], gnames[j], weight=dist_matrix[i][j])

    return nx_mst(G, algorithm="kruskal")


def layout_tree(G: nx.Graph, k: float = 1.0) -> dict:
    """
    Spring layout in which shorter edges (fewer differences) are drawn shorter.

    spring_layout treats edge weights as attraction, so the inverse distance is
    used (scaled so the shortest edge has weight 1); passing the distance itself
    would pull the most distant genotypes closest together.

    Args:
        G: Tree whose edges have a ``weight`` attribute (number of differences).
        k: networkx optimal node distance (``--distances``).

    Returns:
        dict: node -> (x, y) position.
    """
    lengths = [max(d["weight"], 1.0) for _, _, d in G.edges(data=True)]
    shortest = min(lengths, default=1.0)
    for (_, _, d), length in zip(G.edges(data=True), lengths):
        d["attraction"] = shortest / length
    return nx.spring_layout(G, seed=42, weight="attraction", k=k)


def draw_msn(
    G: nx.Graph,
    cluster_map: dict[str, str],
    metadata: pd.DataFrame,
    column: str,
    paths: OutputPaths,
    layout_scale: float = 1.0,
    diameter_scale: float = 0.05,
) -> None:
    """
    Draw the tree with a pie chart per genotype, coloured by one metadata
    column, and save ``<prefix>.<column>.msn.png/svg``.

    Args:
        G: Tree from ``build_msn``.
        cluster_map: {sample: genotype} from ``collapse_sequences``.
        metadata: Metadata indexed by strain.
        column: Metadata column to colour by.
        paths: From ``ensure_output_dirs``.
        layout_scale: networkx spring-layout ``k`` (``--distances``).
        diameter_scale: Radius of the largest pie, in layout units
            (``--diameter``).
    """
    log.info(f"Drawing network for {column}...")
    label_data = categories(metadata[column])
    color_map = category_colors(metadata[column])

    pie_data = defaultdict(lambda: defaultdict(int))
    for strain, gid in cluster_map.items():
        pie_data[gid][label_data[strain]] += 1

    max_total = max((sum(values.values()) for values in pie_data.values()), default=1)

    pos = layout_tree(G, layout_scale)

    # constrained_layout keeps the legend outside the axes inside the figure
    plt.figure(figsize=(12, 12), constrained_layout=True)
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    for node in G.nodes():
        values = pie_data[node]
        total = sum(values.values())
        if total == 0:
            continue
        sizes = [v / total for v in values.values()]
        colors = [color_map[g] for g in values.keys()]
        x, y = pos[node]

        radius = diameter_scale * (total**0.5) / (max_total**0.5)

        plt.gca().pie(
            sizes,
            center=(x, y),
            colors=colors,
            radius=radius,
            wedgeprops=dict(edgecolor="k", linewidth=0.5),
        )
        plt.text(
            x,
            y,
            node,
            fontsize=6,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.5),
        )

    # Each pie() resets the axis limits to its own centre +-1.25, which clips
    # edges; fit the limits to all nodes instead.
    xy = np.array(list(pos.values())).reshape(-1, 2)
    pad = diameter_scale + 0.05 * max(np.ptp(xy, axis=0).max(), 1e-9)
    ax = plt.gca()
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal")

    add_category_legend(ax, color_map, column)
    plt.axis("off")

    save_figure(plt.gcf(), paths, f"{safe_name(column)}.msn")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collapse similar genotypes and draw a minimum spanning tree."
    )
    parser.add_argument("--vcf", required=True, help="VCF file")
    parser.add_argument("--meta", help="Metadata file")
    parser.add_argument(
        "--contract",
        type=non_negative_int,
        default=0,
        help="Collapse samples differing at <= this many sites (default 0)",
    )
    add_out_argument(parser, required=False, default="msn")
    parser.add_argument(
        "--distances",
        type=positive_float,
        default=1.0,
        help="Spring-layout optimal node distance (networkx k)",
    )
    parser.add_argument(
        "--diameter",
        type=positive_float,
        default=0.05,
        help="Radius of the largest pie chart in layout units (default 0.05)",
    )
    args = parser.parse_args()
    setup_logging("network", args)

    paths = ensure_output_dirs(args.out)
    samples, gt_array, _ = read_vcf(args.vcf, fields=["samples", "calldata/GT"])
    gt = alt_allele_counts(gt_array).T  # (n_samples, n_vars), -1 = missing
    report_call_rates(gt, samples)
    _, diffs, compared = distance_matrices(gt)
    require_shared_sites(compared, samples)
    diff_square = squareform(diffs)

    cluster_map, reps = collapse_sequences(samples, diff_square, args.contract)

    summary_out = paths.table("genotype_summary.txt")
    with atomic_write(summary_out) as f:
        gen_to_samples = defaultdict(list)
        for sample, gid in cluster_map.items():
            gen_to_samples[gid].append(sample)
        for gid in sorted(gen_to_samples.keys(), key=lambda x: int(x[1:])):
            f.write(f"#{gid}\n")
            for sample in sorted(gen_to_samples[gid]):
                f.write(f"{sample}\n")
            f.write("\n")
    log.info(f"Saved genotype summary to {summary_out}")

    gnames = [f"G{i + 1}" for i in range(len(reps))]
    dist_matrix = diff_square[np.ix_(reps, reps)]

    if args.meta:
        meta = load_metadata(args.meta, samples)
        safe_names(meta.columns)
    else:
        meta = pd.DataFrame({"group": "default"},
                            index=pd.Index(samples, name="strain"))

    G = build_msn(gnames, dist_matrix)

    for column in meta.columns:
        draw_msn(
            G,
            cluster_map,
            meta,
            column,
            paths,
            layout_scale=args.distances,
            diameter_scale=args.diameter,
        )


if __name__ == "__main__":
    main()
