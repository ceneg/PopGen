"""Shared helpers: metadata loading, genotype conversion and distances."""

import logging

import allel
import numpy as np
import pandas as pd

from popgen.runlog import die

log = logging.getLogger(__name__)

# Variants per block in pairwise_differences (bounds temporary arrays)
PAIRWISE_CHUNK = 50_000


def load_metadata(meta_path: str, samples) -> pd.DataFrame:
    """
    Load the tab-separated metadata and keep the rows of the VCF samples.

    Args:
        meta_path: Metadata file with a ``strain`` column.
        samples: VCF sample names, in VCF order.

    Returns:
        DataFrame indexed by strain, in sample order; samples without metadata
        are dropped (with a warning).
    """
    log.info(f"Loading metadata from {meta_path}...")
    meta = pd.read_csv(meta_path, sep="\t")

    if "strain" not in meta.columns:
        die("Metadata must have 'strain' column.")

    meta = meta.set_index("strain")

    # Filter to samples present in VCF
    samples_set = set(samples)
    meta_present = meta.index.isin(samples_set)
    meta = meta[meta_present]

    # Check for missing samples
    missing = samples_set - set(meta.index)
    if missing:
        log.warning(f"Warning: {len(missing)} samples missing from metadata.")

    # Reorder to match samples list (if present)
    meta = meta.reindex(samples).dropna(how="all")
    return meta


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


def alt_allele_counts(genotypes) -> np.ndarray:
    """
    Alternate-allele counts per call, with -1 for missing calls.

    Haploid VCFs are parsed by scikit-allel as diploid with every second allele
    missing (``0/.``); for those the first allele is used, so a haploid call is
    only missing when its single allele is.

    Args:
        genotypes: ``allel.GenotypeArray`` of shape (n_variants, n_samples, ploidy).

    Returns:
        (n_variants, n_samples) int array.
    """
    g = np.asarray(genotypes)
    if g.ndim == 3 and g.shape[2] > 1 and np.all(g[:, :, 1:] < 0):
        first = g[:, :, 0]
        return np.where(first < 0, -1, (first > 0).astype(first.dtype))
    return allel.GenotypeArray(g).to_n_alt(fill=-1)
