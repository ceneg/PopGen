"""
Linkage disequilibrium decay: r^2 between variant pairs within a window,
averaged in distance bins, smoothed with LOWESS, and the distance at which the
smoothed curve falls to half of its maximum.
"""

import argparse
import logging
import os
import sys

import allel
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import ensure_output_dirs, read_vcf, setup_logging

log = logging.getLogger("ld_decay")


def filter_variants_min_samples(
    gt, min_samples: int = 2, missing_threshold: float = 0.1
) -> tuple[np.ndarray, "allel.GenotypeArray | allel.HaplotypeArray"]:
    """
    Keep variants carried by at least ``min_samples`` samples and with at most
    ``missing_threshold`` missing calls.

    Haploid VCFs (read by scikit-allel as ``0/.``) are converted to a
    ``HaplotypeArray``.

    Returns:
        tuple: (boolean keep mask, genotypes as Genotype- or HaplotypeArray).
    """
    if gt.ndim == 3 and gt.shape[2] > 1 and np.all(gt[:, :, 1] < 0):
        log.info(
            "Detected Haploid data parsed as Diploid (0/.). "
            "Converting to HaplotypeArray."
        )
        gt = allel.HaplotypeArray(gt[:, :, 0])

    if isinstance(gt, allel.GenotypeArray):
        gn = gt.to_n_alt()
        samples_with_alt = np.sum(gn > 0, axis=1)
    else:
        samples_with_alt = np.sum(gt > 0, axis=1)

    pass_min_samples = samples_with_alt >= min_samples

    is_missing = gt.is_missing()
    missing_frac = np.mean(is_missing, axis=1)
    pass_missing = missing_frac <= missing_threshold

    keep = pass_min_samples & pass_missing
    log.info(f"Filtering: Min Samples >= {min_samples}, Missing <= {missing_threshold}")
    log.info(f"Kept {np.sum(keep)} out of {len(keep)} variants.")

    return keep, gt


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

    if isinstance(gt, allel.GenotypeArray):
        gn = gt.to_n_alt(fill=-1)
    elif isinstance(gt, allel.HaplotypeArray):
        gn = gt[:]
    else:
        gn = gt

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
    its maximum, linearly interpolated. None if it never does.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    peak = int(np.argmax(y))
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
    """Mean r^2 per non-empty bin, at the bin centre (columns dist, r2)."""
    filled = np.flatnonzero(n_pairs)
    return pd.DataFrame(
        {
            "dist": filled * bin_size + bin_size / 2,
            "r2": r2_sum[filled] / n_pairs[filled],
        }
    )


def plot_ld_decay(r2_sum, n_pairs, paths, bin_size: int = 100) -> None:
    """
    Plot the binned mean r^2 with the half-decay distance of its LOWESS curve,
    and write the binned values to ``<prefix>.ld_decay.tsv``.
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

    try:
        sx, sy = smooth_lowess(binned["dist"], binned["r2"], frac=0.1)

        max_r2 = binned["r2"].max()
        half_r2 = max_r2 / 2

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
                "Smoothed curve does not decay to half of the maximum r^2 "
                f"(max {max_r2:.3f}, {sy[-1]:.3f} at {sx[-1]:.0f} bp): LD extends "
                "beyond the window; a larger --window may reach it."
            )

    except Exception as e:
        log.info(f"Smoothing failed: {e}")

    plt.xlabel("Distance (bp)")
    plt.ylabel("Linkage Disequilibrium ($r^2$)")
    plt.title("LD Decay")
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{paths['png']}.ld_decay.png")
    plt.savefig(f"{paths['svg']}.ld_decay.svg")
    plt.close()

    binned[["dist", "r2"]].to_csv(f"{paths['tsv']}.ld_decay.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate and visualize LD decay.")
    parser.add_argument("--vcf", required=True, help="Input VCF")
    parser.add_argument("--out", required=True, help="Output prefix")
    parser.add_argument(
        "--window", type=int, default=10000, help="Max distance (bp) for LD calculation"
    )
    parser.add_argument(
        "--bin-size", type=int, default=100, help="Bin size (bp) for plotting"
    )
    parser.add_argument(
        "--min-samples", type=int, default=2, help="Min samples with non-ref allele"
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Subset fraction of variants (random downsampling)",
    )
    args = parser.parse_args()
    setup_logging("ld_decay", args)

    paths = ensure_output_dirs(args.out)
    samples, gt_array, callset = read_vcf(
        args.vcf, fields=["samples", "calldata/GT", "variants/POS", "variants/CHROM"]
    )

    keep, gt = filter_variants_min_samples(gt_array, args.min_samples)
    gt = gt.compress(keep, axis=0)
    pos = callset["variants/POS"][keep]
    chrom = callset["variants/CHROM"][keep]

    if args.subset < 1.0:
        n_keep = int(len(gt) * args.subset)
        rng = np.random.default_rng(42)  # reproducible subset
        indices = np.sort(rng.choice(len(gt), n_keep, replace=False))
        gt = gt.take(indices, axis=0)
        pos = pos[indices]
        chrom = chrom[indices]
        log.info(f"Subsampled to {len(gt)} variants.")

    # Pairs are formed within each contig only
    r2_sum, n_pairs = None, None
    for c in np.unique(chrom):
        mask = chrom == c
        c_gt = gt.compress(mask, axis=0)
        c_pos = pos[mask]

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
