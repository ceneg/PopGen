"""Input and output: VCF reading, output paths, file names and atomic writes."""

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator

import allel
import numpy as np

from popgen.runlog import check_memory, die

log = logging.getLogger(__name__)

# Peak memory of read_vcf + downstream arrays, as a multiple of the VCF size
VCF_MEMORY_FACTOR = 1.0
# Assumed compression ratio of a .vcf.gz / .vcf.bgz, for the memory estimate
GZIP_RATIO = 10


def ensure_biallelic(callset: dict) -> None:
    """Exit with an error if the callset has multi-allelic sites."""
    if "variants/numalt" in callset:
        numalt = callset["variants/numalt"]
        multiallelic_count = np.sum(numalt > 1)
        if multiallelic_count > 0:
            die(f"Found {multiallelic_count} multiallelic sites in VCF. "
                "Please provide a strictly biallelic VCF.")


def read_vcf(vcf_file: str | Path, fields="*", require_gt: bool = True) -> tuple:
    """
    Read a VCF with scikit-allel; exit on unreadable or multi-allelic input.

    Args:
        vcf_file: VCF path (plain or gzip-compressed).
        fields: scikit-allel fields to load (``variants/numalt`` is added so
            that multi-allelic sites are always detected).
        require_gt: Exit if the VCF has no samples, or if genotypes were
            requested (``calldata/GT``) and the VCF has none.

    Returns:
        tuple: (samples, ``allel.GenotypeArray`` or None, callset dict).
    """
    log.info(f"Reading VCF file {vcf_file}...")
    vcf_file = str(vcf_file)
    # Read numalt whenever variants are loaded, so every script refuses
    # multi-allelic input (a samples-only read stays a header-only read)
    if fields != "*":
        fields = list(fields)
        if any(f != "samples" for f in fields) and "variants/numalt" not in fields:
            fields.append("variants/numalt")
        path = Path(vcf_file)
        if fields != ["samples"] and path.exists():
            ratio = GZIP_RATIO if path.name.endswith((".gz", ".bgz")) else 1
            check_memory(VCF_MEMORY_FACTOR * ratio * path.stat().st_size,
                         f"Reading {vcf_file}")
    try:
        callset = allel.read_vcf(vcf_file, fields=fields)
    except Exception as e:
        die(f"could not read VCF: {e}")

    if callset is None:
        die("VCF file seems empty or invalid.")

    if "variants/numalt" in callset:
        ensure_biallelic(callset)

    samples = callset.get("samples", [])
    gt = callset.get("calldata/GT")
    gt = allel.GenotypeArray(gt) if gt is not None else None

    if require_gt:
        if len(samples) == 0:
            die(f"{vcf_file} has no sample columns.")
        if gt is None and (fields == "*" or "calldata/GT" in fields):
            die(f"{vcf_file} has no genotypes (FORMAT field GT).")

    if gt is not None:
        log.info(f"Loaded {len(samples)} samples and {gt.shape[0]} variants.")
    else:
        log.info(f"Loaded {len(samples)} samples.")

    return samples, gt, callset


@dataclass(frozen=True)
class OutputPaths:
    """Output files of a prefix ``DIR/NAME``: ``DIR/results/NAME.*``."""

    results: Path
    name: str

    def table(self, suffix: str) -> Path:
        """``DIR/results/NAME.<suffix>`` (tables, Newick, logs)."""
        return self.results / f"{self.name}.{suffix}"

    def png(self, stem: str) -> Path:
        """``DIR/results/png/NAME.<stem>.png``."""
        return self.results / "png" / f"{self.name}.{stem}.png"

    def svg(self, stem: str) -> Path:
        """``DIR/results/svg/NAME.<stem>.svg``."""
        return self.results / "svg" / f"{self.name}.{stem}.svg"


def ensure_output_dirs(out_prefix: str | Path) -> OutputPaths:
    """
    Create ``DIR/results``, ``DIR/results/png`` and ``DIR/results/svg`` for an
    output prefix ``DIR/NAME``.

    Returns:
        OutputPaths: The output file names of the prefix.
    """
    out_path = Path(out_prefix).absolute()
    paths = OutputPaths(out_path.parent / "results", out_path.name)
    for d in (paths.results, paths.results / "png", paths.results / "svg"):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def safe_name(text) -> str:
    """
    ``text`` made safe as a file-name component: runs of characters other than
    letters, digits, ``_``, ``.`` and ``-`` become ``_`` (``dN/dS`` -> ``dN_dS``).
    """
    return re.sub(r"[^\w.-]+", "_", str(text)).strip("_") or "_"


def safe_names(names) -> dict:
    """
    {name: ``safe_name(name)``}; exits if two names give the same file name.
    """
    safe = {}
    for name in names:
        s = safe_name(name)
        clash = [n for n, t in safe.items() if t == s]
        if clash:
            die(f"'{clash[0]}' and '{name}' give the same file name '{s}'; "
                "rename one of them.")
        safe[name] = s
    return safe


@contextmanager
def atomic_write(path: str | Path, mode: str = "w") -> Iterator[IO]:
    """
    Open ``path.tmp`` for writing and rename it to ``path`` when the block ends
    without an error (a killed run leaves no truncated file under ``path``).
    On an error the temporary file is removed and ``path`` is untouched.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, mode) as f:
            yield f
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_table(df, path: str | Path, **to_csv_kwargs) -> None:
    """``df.to_csv(path, **to_csv_kwargs)``, written atomically."""
    with atomic_write(path) as f:
        df.to_csv(f, **to_csv_kwargs)
