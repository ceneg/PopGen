"""
Pairwise divergence and pairwise differences between samples, with clustered
heatmaps (plain and with a colour bar per metadata column).
"""

import argparse
import logging
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

# seaborn's clustermap warns about large or tied matrices
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="seaborn")

log = logging.getLogger("diversity")

# Distances are per variant site: the VCF has no count of invariant callable
# sites, so per-base-pair nucleotide diversity cannot be derived from it.
ND_LABEL = "Pairwise divergence (per variant site)"
# Number of differing sites between two samples (not Watterson's S, which is
# the number of segregating sites in the whole sample)
SS_LABEL = "Pairwise differences (variant sites)"


def compute_distance_matrices(
    genotypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pairwise distances between samples over the variant sites called in both
    samples of a pair.
    Genotypes shape: (n_variants, n_samples, n_ploidy)

    Returns:
        tuple: (divergence, differences) as condensed arrays: the proportion
        of differing sites (normalised Hamming distance) and the number of
        differing sites (Hamming distance).
    """
    diffs, compared = pairwise_differences(alt_allele_counts(genotypes).T)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = diffs / compared
    return dist, diffs


def plot_heatmap(
    matrix: np.ndarray,
    samples,
    meta: pd.DataFrame,
    title: str,
    paths: dict[str, str],
    suffix: str,
    color_by: str | None = None,
) -> None:
    """
    Clustered heatmap of a condensed distance matrix, saved as PNG and SVG.

    Args:
        matrix: Condensed (``pdist`` order) distances.
        samples: Sample names.
        meta: Metadata indexed by sample.
        title: Colour-bar label.
        paths: From ``ensure_output_dirs``.
        suffix: File-name suffix after the output prefix.
        color_by: Metadata column for a row colour bar, or None.
    """
    df = pd.DataFrame(squareform(matrix), index=samples, columns=samples)

    if color_by and color_by in meta.columns:
        # Create row colors
        lut = dict(
            zip(
                meta[color_by].unique(),
                sns.color_palette("tab20", n_colors=len(meta[color_by].unique())),
            )
        )
        row_colors = meta[color_by].map(lut)
        g = sns.clustermap(
            df,
            row_colors=row_colors,
            cmap="viridis",
            cbar_kws={"label": title},
            figsize=(12, 12),
        )
    else:
        g = sns.clustermap(
            df, cmap="viridis", cbar_kws={"label": title}, figsize=(12, 12)
        )

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.savefig(f"{paths['png']}.{suffix}.png")
    g.savefig(f"{paths['svg']}.{suffix}.svg")
    plt.close(g.fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise divergence and difference matrices with heatmaps."
    )
    parser.add_argument("--vcf", required=True, help="Input VCF file")
    parser.add_argument("--meta", required=True, help="Metadata file")
    parser.add_argument("--out", required=True, help="Output prefix")
    args = parser.parse_args()
    setup_logging("diversity", args)

    paths = ensure_output_dirs(args.out)
    samples, gt, _ = read_vcf(args.vcf, fields=["samples", "calldata/GT"])
    meta = load_metadata(args.meta, samples)

    log.info("Computing distance matrices...")
    divergence, differences = compute_distance_matrices(gt)

    for suffix, matrix, label in (
        ("nd", divergence, ND_LABEL),
        ("ss", differences, SS_LABEL),
    ):
        pd.DataFrame(squareform(matrix), index=samples, columns=samples).to_csv(
            f"{paths['tsv']}.{suffix}.tsv", sep="\t"
        )
        log.info(f"Plotting heatmaps for {paths['base']}.{suffix}...")
        plot_heatmap(matrix, samples, meta, label, paths, suffix)
        for col in meta.columns:
            plot_heatmap(
                matrix, samples, meta, label, paths, f"{suffix}.{col}", color_by=col
            )

    log.info("Diversity analysis complete.")


if __name__ == "__main__":
    main()
