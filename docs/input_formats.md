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
  0/1/2); `selection.py` uses only the first allele of each call (and logs
  how many calls were heterozygous); `recombination.py` refuses heterozygous
  calls.
* One ploidy per VCF. A VCF that mixes haploid calls (read as `0/.`) with
  diploid calls (`0/1`) is refused when the haploid calls are the majority
  (re-call the samples with one ploidy, or split the VCF); otherwise the
  haploid calls are treated as missing and counted in a warning.
* Missing calls (`.`) are allowed. Distances (`diversity.py`, `structure.py`,
  `network.py`) use the sites called in both samples of a pair; PCA replaces
  missing calls by the per-variant mean; `ld_decay.py` drops variants with
  more than 10% missing calls and uses the samples called at both variants;
  `selection.py` turns a missing call into N, which removes that codon;
  `recombination.py` writes it as N. Samples called at fewer than half of
  the sites are listed in a warning; `diversity.py`, `structure.py` and
  `network.py` stop if two samples share no called site (remove failed
  samples from the VCF).
* Records must be sorted by position within each contig (`bcftools sort`):
  `ld_decay.py` and `recombination.py` stop otherwise; `selection.py` sorts.
* Sample names must match the `strain` column of the metadata and, for
  `selection.py`, the tip names of the tree.
* For `selection.py`, `CHROM` and `POS` must refer to the same reference FASTA
  and GFF; indels and multi-base variants are skipped.

The whole VCF is loaded into memory; expect peak memory of the order of the
uncompressed VCF size (each script logs a warning when less is available).

## Metadata (`--meta`: `diversity.py`, `structure.py`, `network.py`)

Tab-separated text with a header line. A column named `strain` holds the
sample names; every other column is a categorical variable, and each one gets
its own coloured plot. Every value is read as text, so numeric sample names
(`101`, `007`) match the VCF exactly and a year stays `2019`. Spaces around
column names and strain names are ignored. A file with only the `strain`
column is valid (no coloured plots).

```
strain	habitat	location	lineage
S01	soil	North	A
S02	soil	South	A
```

The scripts stop if the file cannot be read, has no `strain` column, has an
empty or repeated strain name, or names none of the VCF samples. Samples
missing from the metadata, and empty values, trigger one warning and are
drawn in grey as "unknown" in every plot; metadata rows without a matching VCF
sample are ignored.

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
`<prefix>.upgma.nwk` written by `structure.py` (which quotes names containing
spaces or any of `()[]':;,` in single quotes, as Newick requires). Tips without results are
labelled N/A; samples missing from the tree are still in the tables. Tip
names must be unique. The tree is drawn as given (its topology and branch
lengths); a tree without branch lengths is drawn as a cladogram, with a
warning.
