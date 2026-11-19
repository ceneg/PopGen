"""
Genotype layer: the only module that interprets scikit-allel genotype arrays.

Ploidy, allele counts, per-sample call rates, shared called sites, rows per
contig and pairwise differences. Haploid VCFs are read by scikit-allel as
diploid with every second allele missing (``0/.``, a "half" call).
"""

import logging
from typing import Literal

import allel
import numpy as np

from popgen.runlog import die

log = logging.getLogger(__name__)

# Variants per block in pairwise_differences (bounds temporary arrays)
PAIRWISE_CHUNK = 50_000
# Samples called at fewer than this fraction of sites are reported
LOW_CALL_RATE = 0.5
# Names listed in an error or warning message
MAX_NAMES = 10


def _names(names, limit: int = MAX_NAMES) -> str:
    """Comma-separated names, at most ``limit``, then "and N more"."""
    names = [str(n) for n in names]
    text = ", ".join(names[:limit])
    if len(names) > limit:
        text += f" and {len(names) - limit} more"
    return text


def call_states(gt) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify calls of a (n_variants, n_samples, ploidy) genotype array.

    Args:
        gt: Genotype array; negative allele indices are missing.

    Returns:
        tuple: boolean (half, full) arrays of shape (n_variants, n_samples).
        ``half``: first allele called, every later allele missing (``0/.``);
        ``full``: every allele called.
    """
    g = np.asarray(gt)
    first_called = g[:, :, 0] >= 0
    later = g[:, :, 1:]
    half = first_called & np.all(later < 0, axis=2)
    full = first_called & np.all(later >= 0, axis=2)
    return half, full


def ploidy_kind(gt) -> Literal["haploid", "diploid", "mixed"]:
    """
    ``"haploid"`` if no call is full (also for zero variants, a 2-D array or
    ploidy 1), ``"diploid"`` if no call is half, else ``"mixed"``.
    """
    g = np.asarray(gt)
    if g.ndim < 3 or g.shape[2] == 1:
        return "haploid"
    half, full = call_states(g)
    if not full.any():
        return "haploid"
    if not half.any():
        return "diploid"
    return "mixed"


def alt_allele_counts(gt) -> np.ndarray:
    """
    Alternate-allele counts per call, with -1 for missing calls.

    Haploid data (``0/.``): the first allele is used, so a haploid call is
    only missing when its single allele is. Diploid data: number of
    non-reference alleles, -1 if any allele is missing. Mixed data: exits if
    half calls outnumber full calls (haploid data with stray diploid calls);
    otherwise the diploid rule is used and the half calls, treated as
    missing, are counted in a warning.

    Args:
        gt: Genotype array of shape (n_variants, n_samples, ploidy), or
            (n_variants, n_samples) allele indices.

    Returns:
        (n_variants, n_samples) int array.
    """
    g = np.asarray(gt)
    kind = ploidy_kind(g)
    if kind == "haploid":
        first = g if g.ndim == 2 else g[:, :, 0]
        return np.where(first < 0, -1, (first > 0).astype(first.dtype))
    if kind == "mixed":
        half, full = call_states(g)
        n_half, n_full = int(half.sum()), int(full.sum())
        if n_half > n_full:
            die(f"The VCF mixes ploidies: {n_half} haploid calls (like '0/.') and "
                f"{n_full} diploid calls. Re-call the samples with one ploidy, "
                "or split the VCF by ploidy.")
        log.warning(f"Warning: {n_half} haploid calls ({100 * n_half / half.size:.2g}% "
                    "of all calls) in diploid data are treated as missing.")
    return allel.GenotypeArray(g).to_n_alt(fill=-1)


def haploid_alleles(gt, on_het: Literal["error", "first"]) -> np.ndarray:
    """
    Allele index of the first allele of every call, -1 for missing.

    Args:
        gt: Genotype array (n_variants, n_samples, ploidy) or 2-D allele indices.
        on_het: What to do with heterozygous calls (two different called
            alleles): ``"error"`` exits, ``"first"`` keeps the first allele
            and logs a warning with their number.

    Returns:
        (n_variants, n_samples) int array.
    """
    g = np.asarray(gt)
    if g.ndim == 2:
        return g
    first = g[:, :, 0]
    later = g[:, :, 1:]
    het = (first >= 0) & np.any((later >= 0) & (later != first[:, :, None]), axis=2)
    n_het = int(het.sum())
    if n_het:
        if on_het == "error":
            die(f"{n_het} heterozygous genotype calls. PhiPack needs haplotypes; "
                "phase the data or use haploid calls.")
        log.warning(f"Warning: {n_het} heterozygous calls; the first allele of each "
                    "is used.")
    return first


def report_call_rates(counts, samples, warn_below: float = LOW_CALL_RATE) -> np.ndarray:
    """
    Per-sample call rate; one warning lists the samples below ``warn_below``.

    Args:
        counts: (n_samples, n_variants) allele counts, -1 = missing.
        samples: Sample names.
        warn_below: Call-rate threshold of the warning.

    Returns:
        (n_samples,) fraction of variants called per sample.
    """
    counts = np.asarray(counts)
    if counts.shape[1] == 0:
        return np.zeros(counts.shape[0])
    rate = (counts >= 0).mean(axis=1)
    low = np.flatnonzero(rate < warn_below)
    if len(low):
        listed = _names(f"{samples[i]} ({rate[i]:.1%})" for i in low)
        log.warning(f"Warning: {len(low)} samples are called at fewer than "
                    f"{warn_below:.0%} of the sites: {listed}.")
    return rate


def require_shared_sites(compared, samples) -> None:
    """
    Exit if any pair of samples has no site called in both.

    Args:
        compared: Condensed (``pdist`` order) number of sites called in both
            samples of each pair, from ``pairwise_differences``.
        samples: Sample names.
    """
    compared = np.asarray(compared)
    zero = np.flatnonzero(compared == 0)
    if len(zero) == 0:
        return
    i, j = np.triu_indices(len(samples), k=1)
    involved = np.unique(np.concatenate([i[zero], j[zero]]))
    die(f"{len(zero)} pairs of samples share no called site (samples involved: "
        f"{_names(samples[k] for k in involved)}). Remove samples with failed or "
        "nearly empty genotype calls from the VCF.")


def contig_rows(
    chrom, pos, on_unsorted: Literal["error", "sort"]
) -> dict[str, np.ndarray]:
    """
    Row indices of every contig, contigs in order of first appearance.

    Args:
        chrom: Contig of every variant.
        pos: Position of every variant.
        on_unsorted: If positions decrease within a contig: ``"error"`` exits
            naming the first such contig, ``"sort"`` sorts its rows by
            position (stable).

    Returns:
        dict: {contig: row indices}.
    """
    chrom = np.asarray(chrom)
    pos = np.asarray(pos)
    if len(chrom) == 0:
        return {}
    uniq, first, inverse = np.unique(chrom, return_index=True, return_inverse=True)
    grouped = np.argsort(inverse.ravel(), kind="stable")
    bounds = np.cumsum(np.bincount(inverse.ravel(), minlength=len(uniq)))[:-1]
    per_contig = np.split(grouped, bounds)
    rows = {}
    for k in np.argsort(first):
        r = per_contig[k]
        p = pos[r]
        if len(r) > 1 and np.any(p[1:] < p[:-1]):
            if on_unsorted == "error":
                at = int(np.flatnonzero(p[1:] < p[:-1])[0]) + 1
                die(f"VCF records are not sorted by position: contig {uniq[k]} "
                    f"goes from {p[at - 1]} to {p[at]}. Sort the VCF first "
                    "(for example with bcftools sort).")
            r = r[np.argsort(p, kind="stable")]
        rows[str(uniq[k])] = r
    return rows


def pairwise_differences(
    g2d, chunk: int = PAIRWISE_CHUNK
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pairwise allele-count mismatches between samples, ignoring missing calls.

    Variants are processed in blocks of ``chunk`` so that temporary arrays stay
    small for genome-wide data.

    Args:
        g2d: (n_samples, n_variants) integer array, e.g.
            ``GenotypeArray.to_n_alt(fill=-1).T``. Negative values mark missing calls.
        chunk: Number of variants per block.

    Returns:
        tuple: (diffs, compared), condensed (``scipy.spatial.distance.pdist`` order)
        arrays with the number of differing sites and the number of sites called
        in both samples of each pair.
    """
    g2d = np.asarray(g2d)
    n_samples, n_vars = g2d.shape
    iu = np.triu_indices(n_samples, k=1)
    diffs = np.zeros((n_samples, n_samples))
    compared = np.zeros((n_samples, n_samples))
    values = np.unique(g2d[g2d >= 0]) if n_vars else []

    for start in range(0, n_vars, chunk):
        block = g2d[:, start : start + chunk]
        called = (block >= 0).astype(np.float32)
        both = called @ called.T
        # Pairs called at a site with equal values, summed over allele counts
        same = np.zeros_like(both)
        for v in values:
            is_v = (block == v).astype(np.float32)
            same += is_v @ is_v.T
        compared += both
        diffs += both - same
    return np.rint(diffs[iu]), compared[iu]
