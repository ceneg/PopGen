"""
Pairwise divergence and pairwise differences between samples, with clustered
heatmaps (plain and with a colour bar per metadata column).
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from popgen.cli import add_out_argument
from popgen.genotypes import (
    alt_allele_counts,
    distance_matrices,
    report_call_rates,
    require_shared_sites,
)
from popgen.io import (
    OutputPaths,
    ensure_output_dirs,
    load_metadata,
    read_vcf,
    safe_names,
    write_table,
)
from popgen.plotting import (
    add_category_legend,
    categories,
    category_colors,
    save_figure,
)
from popgen.runlog import die, setup_logging

log = logging.getLogger("diversity")

# Distances are per variant site: the VCF has no count of invariant callable
# sites, so per-base-pair nucleotide diversity cannot be derived from it.
ND_LABEL = "Pairwise divergence (per variant site)"
# Number of differing sites between two samples (not Watterson's S, which is
# the number of segregating sites in the whole sample)
SS_LABEL = "Pairwise differences (variant sites)"


def plot_heatmap(
    matrix: np.ndarray,
    samples,
    meta: pd.DataFrame,
    title: str,
    paths: OutputPaths,
    suffix: str,
    color_by: str | None = None,
) -> None:
    """
    Heatmap of a condensed distance matrix, rows and columns ordered by
    average linkage (UPGMA) on the distances themselves, saved as PNG and SVG.

    Args:
        matrix: Condensed (``pdist`` order) finite distances.
        samples: Sample names.
        meta: Metadata indexed by sample.
        title: Colour-bar label.
        paths: From ``ensure_output_dirs``.
        suffix: File-name part after the output name.
        color_by: Metadata column for a row colour bar and its legend, or None.

    Returns:
        The seaborn ClusterGrid (its figure is already saved and closed).
    """
    df = pd.DataFrame(squareform(matrix), index=samples, columns=samples)
    # clustermap's own clustering would treat each row as a feature vector
    z = linkage(matrix, method="average")
    kwargs = dict(row_linkage=z, col_linkage=z, cmap="viridis",
                  cbar_kws={"label": title}, figsize=(12, 12))

    if color_by and color_by in meta.columns:
        colors = category_colors(meta[color_by])
        row_colors = categories(meta[color_by]).map(colors)
        row_colors.index = df.index
        row_colors.name = color_by
        g = sns.clustermap(df, row_colors=row_colors, **kwargs)
        add_category_legend(g.fig, colors, color_by)
    else:
        g = sns.clustermap(df, **kwargs)

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    # bbox_inches="tight" is the default of seaborn's own Grid.savefig
    save_figure(g.fig, paths, suffix, bbox_inches="tight")
    return g


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise divergence and difference matrices with heatmaps."
    )
    parser.add_argument("--vcf", required=True, help="Input VCF file")
    parser.add_argument("--meta", required=True, help="Metadata file")
    add_out_argument(parser)
    args = parser.parse_args()
    setup_logging("diversity", args)

    paths = ensure_output_dirs(args.out)
    samples, gt, _ = read_vcf(args.vcf, fields=["samples", "calldata/GT"])
    if len(samples) < 2:
        die("diversity needs at least 2 samples.")
    meta = load_metadata(args.meta, samples)
    file_names = safe_names(meta.columns)

    log.info("Computing distance matrices...")
    counts = alt_allele_counts(gt).T  # (n_samples, n_variants)
    report_call_rates(counts, samples)
    divergence, differences, compared = distance_matrices(counts)
    require_shared_sites(compared, samples)

    for suffix, matrix, label in (
        ("nd", divergence, ND_LABEL),
        ("ss", differences, SS_LABEL),
    ):
        write_table(
            pd.DataFrame(squareform(matrix), index=samples, columns=samples),
            paths.table(f"{suffix}.tsv"), sep="\t",
        )
        log.info(f"Plotting heatmaps for {paths.table(suffix)}...")
        plot_heatmap(matrix, samples, meta, label, paths, suffix)
        for col in meta.columns:
            plot_heatmap(
                matrix, samples, meta, label, paths, f"{suffix}.{file_names[col]}",
                color_by=col,
            )

    log.info("Diversity analysis complete.")


if __name__ == "__main__":
    main()
