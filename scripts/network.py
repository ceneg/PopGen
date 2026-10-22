"""
Genotype network: collapse near-identical samples into genotypes, connect the
genotypes by a minimum spanning tree of their pairwise differences, and draw
it with a pie chart of metadata categories at every node.
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.tree import minimum_spanning_tree as nx_mst
from scipy.spatial.distance import squareform

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    alt_allele_counts,
    ensure_output_dirs,
    load_metadata,
    pairwise_differences,
    read_vcf,
    setup_logging,
)

log = logging.getLogger("network")


def collapse_sequences(
    names, genotypes: np.ndarray, threshold: int
) -> tuple[dict[str, str], list[np.ndarray]]:
    """
    Group samples whose genotypes differ at no more than ``threshold`` sites.

    Greedy, in input order: each sample joins the first group whose
    representative (its first member) is within the threshold. Sites missing
    (< 0) in either sample are not counted.

    Args:
        names: Sample names.
        genotypes: (n_samples, n_variants) allele counts, -1 = missing.
        threshold: Largest number of differences within a group.

    Returns:
        tuple: ({sample: "G<n>"}, list of group representative genotypes).
    """
    log.info(f"Clustering sequences with contract threshold {threshold}...")
    clusters = []
    cluster_map = {}

    for name, seq in zip(names, genotypes):
        for idx, rep in enumerate(clusters):
            diff = np.sum((seq != rep) & (seq >= 0) & (rep >= 0))
            if diff <= threshold:
                cluster_map[name] = f"G{idx + 1}"
                break
        else:
            clusters.append(seq)
            cluster_map[name] = f"G{len(clusters)}"

    log.info(f"Identified {len(clusters)} unique genotypes (clusters).")
    return cluster_map, clusters


def compute_distance_matrix(clusters) -> tuple[list[str], np.ndarray]:
    """
    Pairwise differences between group representatives (sites called in both).

    Returns:
        tuple: (group names "G1".., square distance matrix).
    """
    log.info("Computing distance matrix for genotypes...")
    gnames = [f"G{i+1}" for i in range(len(clusters))]
    gseqs = np.array(clusters)

    diffs, _ = pairwise_differences(gseqs)
    dist_matrix = squareform(diffs)
    return gnames, dist_matrix


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


def get_color_map(groups) -> dict:
    """Map categories to tab20 (<= 20 categories) or hsv colours."""
    n = len(groups)
    base_cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
    colors = [base_cmap(i / n) for i in range(n)]
    return dict(zip(groups, colors))


def draw_msn(
    G: nx.Graph,
    cluster_map: dict[str, str],
    metadata: pd.DataFrame,
    column: str,
    paths: dict[str, str],
    layout_scale: float = 1.0,
    diameter_scale: float = 0.001,
) -> None:
    """
    Draw the tree with a pie chart per genotype, coloured by one metadata
    column, and save ``<prefix>.<column>.msn.png/svg``.

    Args:
        G: Tree from ``build_msn``.
        cluster_map: {sample: genotype} from ``collapse_sequences``.
        metadata: Metadata with a ``strain`` column.
        column: Metadata column to colour by.
        paths: From ``ensure_output_dirs``.
        layout_scale: networkx spring-layout ``k`` (``--distances``).
        diameter_scale: Radius of the largest pie, in layout units
            (``--diameter``).
    """
    log.info(f"Drawing network for {column}...")
    metadata = metadata.set_index("strain")
    label_data = metadata[column]

    pie_data = defaultdict(lambda: defaultdict(int))
    for strain, gid in cluster_map.items():
        val = label_data.get(strain, "unknown")
        if pd.isna(val):
            val = "unknown"
        pie_data[gid][val] += 1

    # key=str: metadata values may mix numbers and the "unknown" label
    groups = sorted(set().union(*(d.keys() for d in pie_data.values())), key=str)

    color_map = get_color_map(groups)
    default_color = (0.5, 0.5, 0.5, 1.0)

    def get_color(key):
        return color_map.get(key, default_color)

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
        colors = [get_color(g) for g in values.keys()]
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

    for label, color in color_map.items():
        plt.plot([], [], marker="o", color=color, linestyle="", label=str(label))

    plt.legend(title=column, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.axis("off")

    plt.savefig(f"{paths['png']}.{column}.msn.png")
    plt.savefig(f"{paths['svg']}.{column}.msn.svg")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collapse similar genotypes and draw a minimum spanning tree."
    )
    parser.add_argument("--vcf", required=True, help="VCF file")
    parser.add_argument("--meta", help="Metadata file")
    parser.add_argument(
        "--contract",
        type=int,
        default=0,
        help="Collapse samples differing at <= this many sites (default 0)",
    )
    parser.add_argument("--out", type=str, default="msn", help="Output file prefix")
    parser.add_argument(
        "--distances",
        type=float,
        default=1.0,
        help="Spring-layout optimal node distance (networkx k)",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=0.001,
        help="Radius of the largest pie chart in layout units (default 0.001)",
    )
    args = parser.parse_args()
    setup_logging("network", args)

    paths = ensure_output_dirs(args.out)
    samples, gt_array, _ = read_vcf(args.vcf, fields=["samples", "calldata/GT"])
    gt = alt_allele_counts(gt_array).T  # (n_samples, n_vars), -1 = missing

    cluster_map, clusters = collapse_sequences(samples, gt, args.contract)

    summary_out = f"{paths['tsv']}.genotype_summary.txt"
    with open(summary_out, "w") as f:
        gen_to_samples = defaultdict(list)
        for sample, gid in cluster_map.items():
            gen_to_samples[gid].append(sample)
        for gid in sorted(gen_to_samples.keys(), key=lambda x: int(x[1:])):
            f.write(f"#{gid}\n")
            for sample in sorted(gen_to_samples[gid]):
                f.write(f"{sample}\n")
            f.write("\n")
    log.info(f"Saved genotype summary to {summary_out}")

    gnames, dist_matrix = compute_distance_matrix(clusters)

    if args.meta:
        meta = load_metadata(args.meta, samples).reset_index()
    else:
        meta = pd.DataFrame({"strain": samples, "group": ["default"] * len(samples)})

    G = build_msn(gnames, dist_matrix)

    for column in meta.columns:
        if column == "strain":
            continue

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
