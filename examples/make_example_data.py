#!/usr/bin/env python3
"""
Write a small synthetic dataset for trying out and testing the scripts.

The data are simulated, not real: 24 haploid samples from three lineages, two
contigs with protein-coding genes, and SNPs from lineage-specific mutations,
private mutations and occasional exchange of segments between lineages (so
that LD decays with distance). The output is deterministic (fixed seed).

Files written to ``--outdir`` (default: ``examples/data``):
    example.fna       reference genome (FASTA)
    example.gff       gene annotation (GFF3, gene / mRNA / CDS)
    example.vcf       haploid, biallelic SNPs (VCF 4.2)
    metadata.tsv      sample metadata (strain, habitat, location, lineage)
"""

import argparse
from pathlib import Path

import numpy as np

SEED = 2026
CONTIGS = {"ctg1": 30_000, "ctg2": 20_000}
GENES_PER_CONTIG = 6
N_PER_LINEAGE = 8
LINEAGES = ("A", "B", "C")
SNPS_PER_KB = 12
PRIVATE_FRACTION = 0.3  # SNPs private to one to three samples
SWITCH_MEAN_BP = 4_000  # mean length of a segment copied from one lineage
FOREIGN_SEGMENT = 0.2  # probability that a segment comes from another lineage
MISSING_RATE = 0.005
BASES = np.array(list("ACGT"))
STOPS = {"TAA", "TAG", "TGA"}
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def random_cds(rng: np.random.Generator, n_codons: int) -> str:
    """ATG + random sense codons + a stop codon."""
    codons = ["ATG"]
    while len(codons) < n_codons - 1:
        codon = "".join(rng.choice(BASES, 3))
        if codon not in STOPS:
            codons.append(codon)
    codons.append("TAA")
    return "".join(codons)


def gff_phases(lengths: list[int]) -> list[int]:
    """GFF3 phase of each CDS, given CDS lengths in transcription order."""
    phases, done = [], 0
    for n in lengths:
        phases.append((3 - done % 3) % 3)
        done += n
    return phases


def build_contig(
    rng: np.random.Generator, name: str, length: int
) -> tuple[str, list[str]]:
    """Random contig sequence with genes of two CDS each; returns (seq, GFF lines)."""
    seq = list("".join(rng.choice(BASES, length)))
    gff = []
    slot = length // GENES_PER_CONTIG
    for g in range(GENES_PER_CONTIG):
        gene_id = f"{name}_g{g + 1}"
        cds = random_cds(rng, int(rng.integers(150, 400)))
        cut = int(rng.integers(len(cds) // 3, 2 * len(cds) // 3))
        if cut % 3 == 0:
            cut += 1  # make the second CDS start mid-codon (non-zero phase)
        intron = int(rng.integers(60, 150))
        strand = "+" if g % 2 == 0 else "-"
        # Exon parts in transcription order; minus-strand genes are written
        # reverse-complemented, so their first part lies at the higher position.
        parts = [cds[:cut], cds[cut:]]
        start = g * slot + int(rng.integers(200, 600))
        span = len(cds) + intron
        assert start + span < (g + 1) * slot
        if strand == "+":
            coords = [(start, start + cut), (start + cut + intron, start + span)]
            seq[coords[0][0] : coords[0][1]] = parts[0]
            seq[coords[1][0] : coords[1][1]] = parts[1]
        else:
            rc = [p.translate(COMPLEMENT)[::-1] for p in parts]
            coords = [
                (start + span - cut, start + span),
                (start, start + span - cut - intron),
            ]
            seq[coords[0][0] : coords[0][1]] = rc[0]
            seq[coords[1][0] : coords[1][1]] = rc[1]
        lo, hi = start + 1, start + span  # 1-based inclusive
        gff.append(f"{name}\tsim\tgene\t{lo}\t{hi}\t.\t{strand}\t.\tID=gene-{gene_id}")
        gff.append(
            f"{name}\tsim\tmRNA\t{lo}\t{hi}\t.\t{strand}\t.\t"
            f"ID=rna-{gene_id};Parent=gene-{gene_id}"
        )
        for (a, b), phase in zip(coords, gff_phases([len(p) for p in parts])):
            gff.append(
                f"{name}\tsim\tCDS\t{a + 1}\t{b}\t.\t{strand}\t{phase}\t"
                f"ID=cds-{gene_id};Parent=rna-{gene_id}"
            )
    return "".join(seq), gff


def mosaic_sources(
    rng: np.random.Generator, lineage: int, positions: np.ndarray, length: int
) -> np.ndarray:
    """Lineage that a sample copies at each SNP position (mostly its own)."""
    source = np.empty(len(positions), dtype=int)
    start = 0
    while start < length:
        end = start + int(rng.exponential(SWITCH_MEAN_BP)) + 1
        if rng.random() < FOREIGN_SEGMENT:
            src = int(rng.choice([i for i in range(len(LINEAGES)) if i != lineage]))
        else:
            src = lineage
        source[(positions >= start) & (positions < end)] = src
        start = end
    return source


def simulate_genotypes(
    rng: np.random.Generator, length: int, sample_lineage: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SNP positions (0-based, sorted) and (n_snps, n_samples) 0/1 genotypes."""
    n_snps = int(length / 1000 * SNPS_PER_KB)
    pos = np.sort(rng.choice(np.arange(10, length - 10), n_snps, replace=False))
    n_samples = len(sample_lineage)

    # Alleles of the lineage ancestors: each carries a site with p = 0.35
    templates = (rng.random((n_snps, len(LINEAGES))) < 0.35).astype(int)
    gt = np.empty((n_snps, n_samples), dtype=int)
    for s in range(n_samples):
        src = mosaic_sources(rng, sample_lineage[s], pos, length)
        gt[:, s] = templates[np.arange(n_snps), src]

    private = rng.random(n_snps) < PRIVATE_FRACTION
    for i in np.flatnonzero(private):
        gt[i] = 0
        gt[i, rng.choice(n_samples, int(rng.integers(1, 4)), replace=False)] = 1

    polymorphic = (gt.min(axis=1) == 0) & (gt.max(axis=1) == 1)
    return pos[polymorphic], gt[polymorphic]


def write_vcf(
    path: Path,
    samples: list[str],
    contigs: dict[str, str],
    rng: np.random.Generator,
    sample_lineage: np.ndarray,
) -> None:
    """Simulate SNPs on every contig and write a haploid VCF."""
    lines = [
        "##fileformat=VCFv4.2",
        "##source=examples/make_example_data.py (simulated data)",
    ]
    lines += [f"##contig=<ID={c},length={len(s)}>" for c, s in contigs.items()]
    lines.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    lines.append(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)
    )
    for contig, seq in contigs.items():
        pos, gt = simulate_genotypes(rng, len(seq), sample_lineage)
        calls = gt.astype(str)
        calls[rng.random(calls.shape) < MISSING_RATE] = "."
        for p, row in zip(pos, calls):
            ref = seq[p]
            alt = str(rng.choice([b for b in "ACGT" if b != ref]))
            lines.append(
                f"{contig}\t{p + 1}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(row)
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Output directory (default: examples/data)",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    contigs, gff = {}, ["##gff-version 3"]
    for name, length in CONTIGS.items():
        contigs[name], lines = build_contig(rng, name, length)
        gff += lines

    with open(args.outdir / "example.fna", "w") as f:
        for name, seq in contigs.items():
            f.write(f">{name} simulated contig\n")
            f.writelines(seq[i : i + 80] + "\n" for i in range(0, len(seq), 80))
    (args.outdir / "example.gff").write_text("\n".join(gff) + "\n")

    samples, sample_lineage, meta = [], [], ["strain\thabitat\tlocation\tlineage"]
    habitats, locations = ("soil", "water", "indoor"), ("North", "South")
    for li, lineage in enumerate(LINEAGES):
        for k in range(N_PER_LINEAGE):
            name = f"S{len(samples) + 1:02d}"
            samples.append(name)
            sample_lineage.append(li)
            habitat = habitats[(li + k // 3) % 3]
            location = locations[(li + k) % 2]
            meta.append(f"{name}\t{habitat}\t{location}\t{lineage}")
    (args.outdir / "metadata.tsv").write_text("\n".join(meta) + "\n")

    write_vcf(
        args.outdir / "example.vcf", samples, contigs, rng, np.array(sample_lineage)
    )
    print(f"Wrote example data to {args.outdir}")


if __name__ == "__main__":
    main()
