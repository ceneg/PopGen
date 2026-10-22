# Input formats

Small examples of every format are in [`examples/data/`](../examples/data/)
(simulated data, created by `examples/make_example_data.py`).

## VCF (all scripts)

* VCF 4.x, plain text (`.vcf`) or gzip-compressed (`.vcf.gz`), as read by
  scikit-allel.
* **Biallelic sites only.** Every script stops if any site has more than one
  ALT allele. Filter first, e.g. `bcftools view -m2 -M2 -v snps`.
* Genotypes in the `GT` field. The scripts are written for **haploid** calls
  (`0`, `1`, `.`). Diploid calls are accepted by `diversity.py`,
  `structure.py`, `network.py` and `ld_decay.py` (alternate-allele counts
  0/1/2); `selection.py` uses only the first allele of each call;
  `recombination.py` refuses heterozygous calls.
* Missing calls (`.`) are allowed. Distances (`diversity.py`, `structure.py`,
  `network.py`) use the sites called in both samples of a pair; PCA replaces
  missing calls by the per-variant mean; `ld_decay.py` drops variants with
  more than 10% missing calls and uses the samples called at both variants;
  `selection.py` turns a missing call into N, which removes that codon;
  `recombination.py` writes it as N.
* Sample names must match the `strain` column of the metadata and, for
  `selection.py`, the tip names of the tree.
* For `selection.py`, `CHROM` and `POS` must refer to the same reference FASTA
  and GFF; indels and multi-base variants are skipped.

The whole VCF is loaded into memory; expect peak memory of the order of the
uncompressed VCF size (each script logs a warning when less is available).

## Metadata (`--meta`: `diversity.py`, `structure.py`, `network.py`)

Tab-separated text with a header line. A column named `strain` holds the
sample names; every other column is a categorical variable, and each one gets
its own coloured plot.

```
strain	habitat	location	lineage
S01	soil	North	A
S02	soil	South	A
```

Samples missing from the metadata trigger a warning and have no category in the
plots (`network.py` labels them "unknown"); metadata rows without a matching
VCF sample are ignored.

## Reference genome (`--fasta`: `selection.py`)

FASTA with one record per contig. Record IDs (the text after `>` up to the
first space) must match the VCF `CHROM` values and the GFF sequence IDs.
Lowercase (soft-masked) bases are accepted.

## Annotation (`--gff`: `selection.py`)

GFF3. Only `CDS` lines are used. CDS lines are grouped into coding sequences by
their `Parent` attribute (falling back to `gene`, then `ID`), and the phase
column is respected. NCBI RefSeq GFF3 files work as they are.

## Tree (`--nwk`: `selection.py`)

Newick tree whose tip names are VCF sample names, for example the
`<prefix>.upgma.nwk` written by `structure.py`. Tips without results are
labelled N/A; samples missing from the tree are still in the tables.
