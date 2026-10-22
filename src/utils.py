"""Shared helpers: VCF reading, genotype conversion, distances, logging, paths."""

import logging
import os
import platform
import shutil
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import allel
import numpy as np
import pandas as pd
import psutil

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PACKAGES = ("numpy", "pandas", "scipy", "scikit-allel", "scikit-learn",
                "biopython", "networkx", "statsmodels", "matplotlib", "seaborn")
# Peak memory of read_vcf + downstream arrays, as a multiple of the VCF size
VCF_MEMORY_FACTOR = 1.0
# Variants per block in pairwise_differences (bounds temporary arrays)
PAIRWISE_CHUNK = 50_000


def setup_logging(script: str, args=None, log_dir: str | None = None) -> Path:
    """
    Send log messages to the console and to ``logs/<script>_<timestamp>.log``,
    and record the parameters, Python and package versions in the log file.

    Args:
        script: Script name used for the log file name.
        args: Parsed ``argparse`` namespace to record, or None.
        log_dir: Directory for the log file (default: the ``POPGEN_LOG_DIR``
            environment variable if set, otherwise ``<project>/logs``).

    Returns:
        Path: The log file.
    """
    log_dir = log_dir or os.environ.get("POPGEN_LOG_DIR")
    log_dir = Path(log_dir) if log_dir else PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script}_{stamp}_{os.getpid()}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    to_file = logging.FileHandler(log_file)
    to_file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(console)
    root.addHandler(to_file)

    versions = []
    for pkg in LOG_PACKAGES:
        try:
            versions.append(f"{pkg}={importlib_metadata.version(pkg)}")
        except importlib_metadata.PackageNotFoundError:
            pass
    # Run details go to the log file only
    for line in (
        f"Command: {' '.join(sys.argv)}",
        f"Parameters: {vars(args) if args is not None else {}}",
        f"Python {platform.python_version()}; " + ", ".join(versions),
    ):
        to_file.handle(
            logging.LogRecord(script, logging.INFO, __file__, 0, line, None, None)
        )
    return log_file


def check_memory(required_bytes: float, what: str) -> bool:
    """
    Warn when ``required_bytes`` exceeds the memory currently available.

    Args:
        required_bytes: Estimated peak memory of the next step.
        what: Description of the step, for the message.

    Returns:
        bool: True if enough memory appears to be available.
    """
    available = psutil.virtual_memory().available
    if required_bytes > available:
        log.warning(
            f"Warning: {what} may need ~{required_bytes / 1e9:.1f} GB, "
            f"but only {available / 1e9:.1f} GB of memory is available."
        )
        return False
    return True


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
        log.error("Error: Metadata must have 'strain' column.")
        sys.exit(1)

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


def ensure_biallelic(callset: dict) -> None:
    """Exit with an error if the callset has multi-allelic sites."""
    if "variants/numalt" in callset:
        numalt = callset["variants/numalt"]
        multiallelic_count = np.sum(numalt > 1)
        if multiallelic_count > 0:
            log.error(f"Error: Found {multiallelic_count} multiallelic sites in VCF.")
            log.error("Please provide a strictly biallelic VCF.")
            sys.exit(1)


def read_vcf(vcf_file: str, fields="*") -> tuple:
    """
    Read a VCF with scikit-allel; exit on unreadable or multi-allelic input.

    Args:
        vcf_file: VCF path (plain or gzip-compressed).
        fields: scikit-allel fields to load (``variants/numalt`` is added so
            that multi-allelic sites are always detected).

    Returns:
        tuple: (samples, ``allel.GenotypeArray`` or None, callset dict).
    """
    log.info(f"Reading VCF file {vcf_file}...")
    # Read numalt whenever variants are loaded, so every script refuses
    # multi-allelic input (a samples-only read stays a header-only read)
    if fields != "*":
        fields = list(fields)
        if any(f != "samples" for f in fields) and "variants/numalt" not in fields:
            fields.append("variants/numalt")
        if fields != ["samples"] and os.path.exists(vcf_file):
            check_memory(VCF_MEMORY_FACTOR * os.path.getsize(vcf_file),
                         f"Reading {vcf_file}")
    try:
        callset = allel.read_vcf(vcf_file, fields=fields)
    except Exception as e:
        log.error(f"Error reading VCF: {e}")
        sys.exit(1)

    if callset is None:
        log.error("Error: VCF file seems empty or invalid.")
        sys.exit(1)

    if "variants/numalt" in callset:
        ensure_biallelic(callset)

    samples = callset.get("samples", [])
    gt = callset.get("calldata/GT")
    gt = allel.GenotypeArray(gt) if gt is not None else None

    if gt is not None:
        log.info(f"Loaded {len(samples)} samples and {gt.shape[0]} variants.")
    else:
        log.info(f"Loaded {len(samples)} samples.")

    return samples, gt, callset


def ensure_output_dirs(out_prefix: str) -> dict[str, str]:
    """
    Create ``DIR/results``, ``DIR/results/png`` and ``DIR/results/svg`` for an
    output prefix ``DIR/NAME``.

    Returns:
        dict: Path prefixes ``base`` and ``tsv`` (``DIR/results/NAME``), ``png``
        and ``svg`` (``DIR/results/png/NAME``, ``DIR/results/svg/NAME``).
    """
    out_path = os.path.abspath(out_prefix)
    dirname = os.path.dirname(out_path)
    basename = os.path.basename(out_path)

    results_dir = os.path.join(dirname, "results")
    png_dir = os.path.join(results_dir, "png")
    svg_dir = os.path.join(results_dir, "svg")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    return {
        "base": os.path.join(results_dir, basename),
        "png": os.path.join(png_dir, basename),
        "svg": os.path.join(svg_dir, basename),
        "tsv": os.path.join(results_dir, basename),
    }


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


def find_phi() -> str | None:
    """
    Path of PhiPack's ``Phi`` program: the one next to the running Python
    interpreter (installed into the virtual environment by
    ``tools/install_phipack.py``, or a conda environment's), else ``Phi`` on
    the PATH. None if neither exists.
    """
    local = Path(sys.executable).parent / "Phi"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return shutil.which("Phi")
