"""
Population structure: PCA, DAPC (k-means clusters chosen by BIC, then LDA) and
a UPGMA tree written as Newick (the tree input of selection.py).
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import ClusterNode, dendrogram, linkage, to_tree
from scipy.spatial.distance import squareform
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from popgen.cli import add_out_argument
from popgen.genotypes import (
    alt_allele_counts,
    distance_matrices,
    report_call_rates,
    require_shared_sites,
)
from popgen.io import (
    atomic_write,
    ensure_output_dirs,
    load_metadata,
    read_vcf,
    safe_names,
    write_table,
)
from popgen.plotting import (
    UNKNOWN,
    add_category_legend,
    categories,
    category_colors,
    save_figure,
)
from popgen.runlog import die, setup_logging

log = logging.getLogger("structure")

ELLIPSE_N_STD = float(np.sqrt(chi2.ppf(0.95, df=2)))
# DAPC: PCs kept up to this cumulative explained variance (adegenet advises
# against retaining near-zero-variance PCs, which make LDA overfit)
DAPC_PCA_VARIANCE = 0.9
DAPC_MAX_K = 10
# Smallest cluster accepted; singleton/clonal clusters have no within-group
# variance and make the discriminant axes degenerate
DAPC_MIN_CLUSTER = 3
# Newick names with any of these characters (or whitespace) are quoted
NEWICK_SPECIAL = set("()[]':;,")


def impute_missing(gt_matrix: np.ndarray) -> np.ndarray:
    """
    Replace missing calls (negative values) with the per-variant mean
    alternate-allele count, the usual imputation for genotype PCA.
    """
    x = gt_matrix.astype(float)
    missing = x < 0
    if missing.any():
        x[missing] = np.nan
        col_mean = np.nan_to_num(np.nanmean(x, axis=0))
        x[missing] = np.take(col_mean, np.nonzero(missing)[1])
    return x


def add_confidence_ellipse(ax, x, y, color="blue") -> tuple[float, float]:
    """
    Draw the 95% coverage ellipse of a bivariate normal fitted to (x, y), for
    groups of at least three points.

    Returns:
        tuple: The group centroid (mean x, mean y).
    """
    if len(x) < 3:
        return np.mean(x), np.mean(y)

    cov = np.cov(x, y)

    scale_x = np.sqrt(cov[0, 0])
    scale_y = np.sqrt(cov[1, 1])
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    if cov[0, 0] <= 0 or cov[1, 1] <= 0:
        return mean_x, mean_y

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    if np.isnan(pearson):
        return mean_x, mean_y
    pearson = float(np.clip(pearson, -1.0, 1.0))  # rounding on collinear groups

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # Radius (in SDs) enclosing 95% of a bivariate normal: sqrt(chi2_2(0.95))
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2 * ELLIPSE_N_STD,
        height=ell_radius_y * 2 * ELLIPSE_N_STD,
        facecolor="none",
        edgecolor=color,
        linewidth=1.5,
        alpha=1.0,
    )

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    return mean_x, mean_y


def dapc_n_pcs(explained_variance_ratio: np.ndarray, n_samples: int) -> int:
    """
    Number of PCs retained for DAPC: enough to explain DAPC_PCA_VARIANCE of the
    variance (all computed PCs if they explain less), at least 2 and at most
    n_samples // 3.
    """
    cum = np.cumsum(explained_variance_ratio)
    n = int(np.searchsorted(cum, min(DAPC_PCA_VARIANCE, cum[-1]) - 1e-12) + 1)
    return int(np.clip(n, 2, max(2, n_samples // 3)))


def choose_dapc_clusters(
    coords: np.ndarray, max_k: int = DAPC_MAX_K, min_size: int = DAPC_MIN_CLUSTER
) -> tuple[np.ndarray | None, int]:
    """
    K-means clustering for DAPC with k chosen by BIC (as in adegenet's
    find.clusters), among k = 3..max_k whose clusters all have at least
    ``min_size`` members. If no k qualifies (e.g. lone outlier samples),
    k = 3 is used.

    Args:
        coords: (n_samples, n_pcs) PCA coordinates.
        max_k: Largest k tried.
        min_size: Smallest acceptable cluster size.

    Returns:
        tuple: (labels, k), or (None, 0) if there are fewer than 4 distinct
        samples.
    """
    n = len(coords)
    n_distinct = len(np.unique(np.round(coords, 10), axis=0))
    best = (None, 0, np.inf)
    fallback = None
    for k in range(3, min(max_k, n_distinct - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(coords)
        if fallback is None:
            fallback = (km.labels_, k)
        if np.bincount(km.labels_).min() < min_size:
            continue
        bic = n * np.log(max(km.inertia_, 1e-300) / n) + k * np.log(n)
        if bic < best[2]:
            best = (km.labels_, k, bic)
    if best[0] is None and fallback is not None:
        log.warning(
            f"Warning: no k gives clusters of >= {min_size} samples; using k = 3."
        )
        return fallback
    return best[0], best[1]


def scatter_groups(df, x, y, hue, color_map, paths, stem, title=None,
                   ellipses=True, label_centroids=True, alpha=0.6,
                   legend_title=None):
    """
    Scatter plot coloured by ``hue``, optionally with a 95% ellipse and a
    centroid marker for every group with more than two members (not for
    ``UNKNOWN``); saved as ``<stem>.png/svg``.

    Args:
        legend_title: Legend title (default: ``hue``).
    """
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="datalim")

    sns.scatterplot(
        x=x, y=y, hue=hue, data=df, palette=color_map, alpha=alpha, ax=ax,
        hue_order=list(color_map), legend=False,
    )

    if ellipses:
        for val, col in color_map.items():
            if val == UNKNOWN:
                continue
            sub = df[df[hue] == val]
            if len(sub) > 2:
                mx, my = add_confidence_ellipse(ax, sub[x], sub[y], color=col)
                ax.scatter(
                    mx,
                    my,
                    marker="X",
                    s=200,
                    color=col,
                    edgecolors="black",
                    linewidth=1.5,
                )
                if label_centroids:
                    ax.text(
                        mx,
                        my,
                        str(val),
                        fontsize=12,
                        fontweight="bold",
                        color="black",
                        ha="center",
                        va="center",
                    )

    add_category_legend(ax, color_map, legend_title or hue)
    if title:
        plt.title(title)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_figure(plt.gcf(), paths, stem, bbox_inches="tight")


def plot_by_metadata(df, x, y, meta, paths, prefix, label) -> None:
    """
    Annotated (ellipses) and simple scatter plots for every metadata column.
    Samples without a value are drawn in grey as ``UNKNOWN``.
    """
    file_names = safe_names(meta.columns)
    for column in meta.columns:
        # A separate frame: a column named like a coordinate must not replace it
        data = df[[x, y]].assign(_category=categories(meta[column]).to_numpy())
        color_map = category_colors(meta[column])
        stem = f"{prefix}.{file_names[column]}"

        scatter_groups(
            data, x, y, "_category", color_map, paths, f"{stem}.annotated",
            title=f"{label} - {column} (with 95% Ellipses)", legend_title=column,
        )
        scatter_groups(
            data, x, y, "_category", color_map, paths, stem,
            ellipses=False,
            alpha=0.5,
            legend_title=column,
        )


def plot_pca(coords, samples, meta, paths) -> None:
    """
    Write the PCA coordinates and plot PC1 vs PC2 (plain and per metadata
    column); with a single component only the coordinates are written.
    """
    n_components = coords.shape[1]
    cols = [f"PC{i+1}" for i in range(n_components)]
    df = pd.DataFrame(coords, columns=cols, index=samples)
    write_table(df, paths.table("pca.tsv"), sep="\t")
    if n_components < 2:
        log.warning("WARNING: only one principal component; PCA plots skipped.")
        return

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="PC1", y="PC2", data=df, alpha=0.5)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    save_figure(plt.gcf(), paths, "pca", bbox_inches="tight")

    plot_by_metadata(df, "PC1", "PC2", meta, paths, "pca", "PCA")


def plot_dapc(coords, samples, meta, paths, groups=None) -> None:
    """Write the DAPC coordinates and plot them by cluster and metadata column."""
    df = pd.DataFrame(coords, columns=["DAPC1", "DAPC2"], index=samples)
    write_table(df, paths.table("dapc.tsv"), sep="\t")

    if groups is not None:
        unique_grps = np.unique(groups)
        palette = sns.color_palette("tab20", len(unique_grps))
        grp_colors = dict(zip(unique_grps, palette))
        scatter_groups(
            df.assign(_group=groups), "DAPC1", "DAPC2", "_group", grp_colors,
            paths, "dapc.clusters", label_centroids=False, legend_title="Group",
        )

    plot_by_metadata(df, "DAPC1", "DAPC2", meta, paths, "dapc", "DAPC")


def newick_label(name) -> str:
    """
    Newick tip label: quoted in single quotes (internal quotes doubled) if the
    name contains whitespace or any of ``()[]':;,``, otherwise unchanged.
    """
    name = str(name)
    if any(c.isspace() or c in NEWICK_SPECIAL for c in name):
        return "'" + name.replace("'", "''") + "'"
    return name


def to_newick(root: ClusterNode, leaf_names) -> str:
    """
    Newick string of a SciPy cluster tree (without the final ``;``).

    Linkage heights are distances between clusters; UPGMA node heights are half
    of them, so that the path between two leaves equals their distance. Branch
    lengths keep 10 significant digits. Built with an explicit stack, so deep
    (ladder-shaped) trees do not hit the recursion limit.
    """
    parts = []
    stack = [(root, root.dist)]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            parts.append(item)
            continue
        node, parent_height = item
        length = format((parent_height - node.dist) / 2, ".10g")
        if node.is_leaf():
            parts.append(f"{newick_label(leaf_names[node.id])}:{length}")
        else:
            parts.append("(")
            stack += [f"):{length}", (node.get_right(), node.dist), ",",
                      (node.get_left(), node.dist)]
    return "".join(parts)


def plot_tree(dist_matrix: np.ndarray, samples, paths) -> None:
    """
    UPGMA (average linkage) tree of a square or condensed distance matrix:
    dendrogram figure and ``<prefix>.upgma.nwk``.
    """
    if dist_matrix.ndim == 2 and dist_matrix.shape[0] == dist_matrix.shape[1]:
        condensed_dist = squareform(dist_matrix)
    else:
        condensed_dist = dist_matrix

    linkage_matrix = linkage(condensed_dist, method="average")

    plt.figure(figsize=(10, max(5, 0.2 * len(samples))))
    dendrogram(linkage_matrix, labels=samples, orientation="left", leaf_font_size=8)
    plt.title("UPGMA Tree (Hamming Distance)")
    plt.xlabel("Distance")
    plt.tight_layout()
    save_figure(plt.gcf(), paths, "upgma", bbox_inches="tight")

    tree = to_tree(linkage_matrix, rd=False)
    with atomic_write(paths.table("upgma.nwk")) as f:
        f.write(to_newick(tree, samples) + ";")


def run_dapc(pca, pca_coords, samples, meta, paths):
    """
    DAPC: k-means clusters on the leading PCs (k by BIC), then LDA; writes the
    coordinates and plots. Never exits: when DAPC is impossible (fewer than 2
    PCs, fewer than 4 distinct samples, or LDA fails on degenerate clusters)
    it logs a WARNING and returns None.

    Returns:
        The cluster labels, or None if DAPC was skipped.
    """
    log.info("Performing DAPC (PCA + LDA with KMeans)...")
    if pca_coords.shape[1] < 2:
        log.warning("WARNING: DAPC skipped: it needs at least 2 principal "
                    "components.")
        return None
    n_pcs = dapc_n_pcs(pca.explained_variance_ratio_, len(samples))
    dapc_input = pca_coords[:, :n_pcs]
    grp, n_clusters = choose_dapc_clusters(dapc_input)
    if grp is None:
        log.warning("WARNING: DAPC skipped: it needs at least 4 genetically "
                    "distinct samples.")
        return None
    log.info(
        f"DAPC: {n_pcs} PCs ({pca.explained_variance_ratio_[:n_pcs].sum():.1%} "
        f"of variance), k = {n_clusters} clusters by BIC, "
        f"sizes {np.bincount(grp).tolist()}"
    )

    # Shrinkage keeps the within-group covariance invertible for tight clusters
    lda = LinearDiscriminantAnalysis(n_components=2, solver="eigen", shrinkage="auto")
    try:
        dapc_coords = lda.fit_transform(dapc_input, grp)
    except (np.linalg.LinAlgError, ValueError) as e:  # e.g. clonal clusters
        log.warning(f"WARNING: DAPC skipped: LDA failed ({e}).")
        return None
    plot_dapc(dapc_coords, samples, meta, paths, groups=grp)
    return grp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA, DAPC and UPGMA tree of SNP genotypes."
    )
    parser.add_argument("--vcf", required=True, help="Input VCF file")
    parser.add_argument("--meta", required=True, help="Metadata file")
    add_out_argument(parser)
    args = parser.parse_args()
    setup_logging("structure", args)

    paths = ensure_output_dirs(args.out)
    samples, gt_array, _ = read_vcf(args.vcf, fields=["samples", "calldata/GT"])
    meta = load_metadata(args.meta, samples)
    safe_names(meta.columns)  # exits now if two columns give one file name

    gt_matrix = alt_allele_counts(gt_array).T  # (n_samples, n_variants), -1 = missing
    if len(samples) < 2:
        die("structure needs at least 2 samples.")
    report_call_rates(gt_matrix, samples)

    # The tree does not depend on PCA or DAPC (and is selection.py's input),
    # so it is built first; a pair without shared sites stops before any output
    log.info("Constructing UPGMA tree...")
    divergence, _, compared = distance_matrices(gt_matrix)
    require_shared_sites(compared, samples)
    plot_tree(divergence, samples, paths)

    log.info("Performing PCA...")
    n_components = min(10, *gt_matrix.shape)
    pca = PCA(n_components=n_components, random_state=42)
    pca_coords = pca.fit_transform(impute_missing(gt_matrix))
    plot_pca(pca_coords, samples, meta, paths)

    run_dapc(pca, pca_coords, samples, meta, paths)

    log.info("Structure analysis complete.")


if __name__ == "__main__":
    main()
