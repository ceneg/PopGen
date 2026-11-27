"""
Linkage disequilibrium decay: r^2 between variant pairs within a window,
averaged in distance bins, smoothed with LOWESS, and the distance at which the
smoothed curve falls to half of its maximum.
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from popgen.cli import add_out_argument, fraction, non_negative_int, positive_int
from popgen.genotypes import alt_allele_counts, contig_rows
from popgen.io import ensure_output_dirs, read_vcf, write_table
from popgen.plotting import save_figure
from popgen.runlog import setup_logging

log = logging.getLogger("ld_decay")


def filter_variants_min_samples(
    gt, min_samples: int = 2, missing_threshold: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep variants carried by at least ``min_samples`` samples and with at most
    ``missing_threshold`` missing calls.

    Returns:
        tuple: (boolean keep mask, (n_variants, n_samples) allele counts from
        ``genotypes.alt_allele_counts``, -1 = missing).
    """
    gn = alt_allele_counts(gt)
    pass_min_samples = (gn > 0).sum(axis=1) >= min_samples
    pass_missing = (gn < 0).mean(axis=1) <= missing_threshold

    keep = pass_min_samples & pass_missing
    log.info(f"Filtering: Min Samples >= {min_samples}, Missing <= {missing_threshold}")
    log.info(f"Kept {np.sum(keep)} out of {len(keep)} variants.")

    return keep, gn


def pairwise_r2(target, neighbors) -> np.ndarray:
    """
    Squared Pearson correlation between one variant and each neighbour,
    using only samples called (>= 0) at both variants.

    Args:
        target: (n_samples,) allele counts, negative = missing.
        neighbors: (n_neighbors, n_samples) allele counts, negative = missing.

    Returns:
        (n_neighbors,) array of r^2; NaN where either variant is monomorphic
        among the shared samples.
    """
    target = np.asarray(target, dtype=float)
    neighbors = np.asarray(neighbors, dtype=float)
    valid = (target >= 0)[None, :] & (neighbors >= 0)
    t = np.where(valid, target[None, :], 0.0)
    u = np.where(valid, neighbors, 0.0)

    n = valid.sum(axis=1)
    sx, sy = t.sum(axis=1), u.sum(axis=1)
    cov = n * (t * u).sum(axis=1) - sx * sy
    var_x = n * (t * t).sum(axis=1) - sx**2
    var_y = n * (u * u).sum(axis=1) - sy**2

    denom = var_x * var_y
    ok = denom > 1e-12
    r2 = np.full(len(neighbors), np.nan)
    r2[ok] = cov[ok] ** 2 / denom[ok]
    return r2


def calculate_windowed_ld(
    gt, pos: np.ndarray, window_bp: int, bin_size: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pairwise r^2 for variants within window_bp, accumulated into distance bins
    (memory stays proportional to the number of bins, not of pairs).

    Bin k holds distances in (k * bin_size, (k + 1) * bin_size].

    Returns:
        tuple: (r2_sum, n_pairs) arrays indexed by bin.
    """
    log.info(f"Calculating pairwise LD (window={window_bp} bp)...")
    gn = np.asarray(gt)
    pos = np.asarray(pos, dtype=np.int64)  # VCF POS is int32: pos + window overflows

    n_bins = max(1, -(-int(window_bp) // bin_size))
    r2_sum = np.zeros(n_bins)
    n_pairs = np.zeros(n_bins, dtype=np.int64)

    n_vars = len(pos)

    for i in range(n_vars):
        start_pos = pos[i]
        end_limit = start_pos + window_bp
        idx_end = np.searchsorted(pos, end_limit, side="right")

        if idx_end <= i + 1:
            continue

        r2 = pairwise_r2(gn[i], gn[i + 1 : idx_end])
        keep = ~np.isnan(r2)
        if not np.any(keep):
            continue

        dist = pos[i + 1 : idx_end][keep] - start_pos
        bins = np.clip((dist - 1) // bin_size, 0, n_bins - 1)
        r2_sum += np.bincount(bins, weights=r2[keep], minlength=n_bins)
        n_pairs += np.bincount(bins, minlength=n_bins)

        if i % 1000 == 0 and i > 0:
            log.info(f"Processed {i}/{n_vars} variants...")

    log.info("Calculation complete.")
    return r2_sum, n_pairs


def half_decay_distance(x, y, half: float) -> float | None:
    """
    Distance at which the curve (x ascending) first falls to ``half`` after
    its maximum, linearly interpolated. None if it never does, or if the
    maximum is not above ``half`` (nothing to decay from; interpolating there
    would extrapolate on the rising side of the peak).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return None
    peak = int(np.argmax(y))
    if y[peak] <= half:
        return None
    below = np.nonzero(y[peak:] <= half)[0]
    if len(below) == 0:
        return None
    j = peak + below[0]
    if j == 0 or y[j] == y[j - 1]:
        return float(x[j])
    return float(x[j - 1] + (half - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1]))


def smooth_lowess(x, y, frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """LOWESS smooth of y on x; returns the sorted x and smoothed y."""
    z = sm.nonparametric.lowess(y, x, frac=frac)
    return z[:, 0], z[:, 1]


def binned_means(r2_sum, n_pairs, bin_size: int = 100) -> pd.DataFrame:
    """
    Mean r^2 per non-empty bin, at the bin centre (columns dist, r2, n_pairs:
    the number of SNP pairs in the bin).
    """
    filled = np.flatnonzero(n_pairs)
    return pd.DataFrame(
        {
            "dist": filled * bin_size + bin_size / 2,
            "r2": r2_sum[filled] / n_pairs[filled],
            "n_pairs": np.asarray(n_pairs)[filled].astype(np.int64),
        }
    )


def plot_ld_decay(r2_sum, n_pairs, paths, bin_size: int = 100) -> None:
    """
    Plot the binned mean r^2 with the half-decay distance of its LOWESS curve,
    and write the binned values to ``<prefix>.ld_decay.tsv``.

    The threshold is half of the smoothed curve's maximum, so one sparse bin
    (few pairs, r^2 near 1) cannot lift it above the curve.
    """
    log.info("Binning and plotting...")

    binned = binned_means(r2_sum, n_pairs, bin_size)
    max_dist = binned["dist"].max() + bin_size / 2

    plt.figure(figsize=(10, 6))

    plt.scatter(
        binned["dist"],
        binned["r2"],
        alpha=0.5,
        color="gray",
        s=10,
        label="Mean $r^2$ (Binned)",
    )

    if len(binned) < 3:
        log.info(f"Too few bins to smooth ({len(binned)}); no half-decay distance.")
    else:
        sx, sy = smooth_lowess(binned["dist"], binned["r2"], frac=0.1)

        max_r2 = float(sy.max())
        half_r2 = max_r2 / 2
        log.info(f"Maximum r^2: smoothed {max_r2:.3f}, raw bins "
                 f"{binned['r2'].max():.3f}")

        intersect_dist = half_decay_distance(sx, sy, half_r2)

        plt.axhline(half_r2, color="blue", linestyle="--", alpha=0.7)

        if intersect_dist is not None and 0 < intersect_dist < max_dist:
            plt.axvline(intersect_dist, color="blue", linestyle="--", alpha=0.7)
            plt.text(
                intersect_dist,
                half_r2 + 0.05,
                f"{intersect_dist:.0f} bp",
                color="blue",
                fontweight="bold",
            )
            log.info(f"Half-decay distance: {intersect_dist:.0f} bp")
        else:
            log.info(
                "Smoothed curve does not decay to half of its maximum r^2 "
                f"(max {max_r2:.3f}, {sy[-1]:.3f} at {sx[-1]:.0f} bp): LD extends "
                "beyond the window; a larger --window may reach it."
            )

    plt.xlabel("Distance (bp)")
    plt.ylabel("Linkage Disequilibrium ($r^2$)")
    plt.title("LD Decay")
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(plt.gcf(), paths, "ld_decay")

    write_table(binned[["dist", "r2", "n_pairs"]], paths.table("ld_decay.tsv"),
                sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate and visualize LD decay.")
    parser.add_argument("--vcf", required=True, help="Input VCF")
    add_out_argument(parser)
    parser.add_argument(
        "--window", type=positive_int, default=10000,
        help="Max distance (bp) for LD calculation",
    )
    parser.add_argument(
        "--bin-size", type=positive_int, default=100, help="Bin size (bp) for plotting"
    )
    parser.add_argument(
        "--min-samples", type=non_negative_int, default=2,
        help="Min samples with non-ref allele",
    )
    parser.add_argument(
        "--subset",
        type=fraction,
        default=1.0,
        help="Fraction (0-1] of variants to keep, drawn at random (seed 42); "
        "unlike recombination.py --subset, which keeps every n-th SNP",
    )
    args = parser.parse_args()
    setup_logging("ld_decay", args)

    paths = ensure_output_dirs(args.out)
    samples, gt_array, callset = read_vcf(
        args.vcf, fields=["samples", "calldata/GT", "variants/POS", "variants/CHROM"]
    )

    keep, gn = filter_variants_min_samples(gt_array, args.min_samples)
    gn = gn[keep]
    pos = callset["variants/POS"][keep]
    chrom = callset["variants/CHROM"][keep]

    if args.subset < 1.0:
        n_keep = int(len(gn) * args.subset)
        rng = np.random.default_rng(42)  # reproducible subset
        indices = np.sort(rng.choice(len(gn), n_keep, replace=False))
        gn = gn[indices]
        pos = pos[indices]
        chrom = chrom[indices]
        log.info(f"Subsampled to {len(gn)} variants.")

    # Pairs are formed within each contig only (contigs in VCF order)
    r2_sum, n_pairs = None, None
    for c, rows in contig_rows(chrom, pos, "error").items():
        c_gt = gn[rows]
        c_pos = pos[rows]

        if len(c_pos) < 2:
            continue

        log.info(f"Processing Chromosome {c} ({len(c_pos)} variants)...")
        s, n = calculate_windowed_ld(c_gt, c_pos, args.window, args.bin_size)
        r2_sum = s if r2_sum is None else r2_sum + s
        n_pairs = n if n_pairs is None else n_pairs + n

    if n_pairs is not None and n_pairs.sum():
        log.info(f"Total pairs calculated: {n_pairs.sum()}")
        plot_ld_decay(r2_sum, n_pairs, paths, args.bin_size)
    else:
        log.info("No LD pairs found.")


if __name__ == "__main__":
    main()
