# Changelog

## 1.10 (2026-11-19)

Results of earlier versions differ where marked **(results change)**; re-run
analyses whose results you report.

Methods and results:

* `selection.py` **(results change)**: codon pairs that involve a stop codon
  are no longer counted (no sites, no differences; before, they counted half
  their sites). dN/dS is `inf` when dS = 0 < dN (before: NaN), for genes and
  genome-wide alike.
* `selection.py` **(results change)**: per-gene rows are in GFF order on
  every run (before, the order was random for more than 20 genes).
* `selection.py`: the tree figures draw the supplied tree as given (its
  topology and branch lengths; before, tips were re-clustered by UPGMA,
  which changed the topology of trees that are not ultrametric). The files
  are named `PREFIX.dN.tree.*`, `PREFIX.dS.tree.*` and `PREFIX.dN_dS.tree.*`
  (before: `.upgma.`).
* `ld_decay.py` **(results change)**: the half-decay distance is where the
  smoothed curve falls to half of its own maximum (before: half of the
  largest raw bin mean, so one sparse bin could give an extrapolated
  distance). No distance is reported when the curve's peak is not above
  half. The LD table gains an `n_pairs` column.
* `diversity.py` **(figures change)**: heatmap dendrograms are average-linkage
  (UPGMA) trees of the distance matrix (before: seaborn clustered the matrix
  rows as Euclidean vectors), with a legend for the metadata colour bar.
* `structure.py`: the UPGMA Newick quotes sample names containing spaces or
  `()[]':;,`, and keeps 10 significant digits in branch lengths (before:
  6 decimals, which rounded very short branches to zero). The tree is written
  before PCA and DAPC; when DAPC is impossible it is skipped with a warning
  instead of stopping the run.
* `recombination.py`: a failed Phi run (for example killed for lack of
  memory) is logged, counted in the summary ("Phi runs failed") and gives
  exit status 2 after all outputs are written; when no contig gives a PHI
  p-value the summary says "not tested" (before: "not rejected"). A
  non-finite p-value printed by Phi is treated as missing (before, it set
  every larger Holm-adjusted p-value to 1). For PHI by permutation,
  `phi_mean` and `phi_z` are the permutation values (before: NA).
* `recombination.py --resume` reuses a Phi log only for the same alignment
  (each log starts with a checksum of the contig's genotypes) and, for a
  permutation test, only with at least as many permutations; logs of 1.01
  are not reused. Contigs whose names map to the same file name get
  separate logs.
* Figures **(figures change)**: a metadata category has the same colour in
  every figure of `diversity.py`, `structure.py` and `network.py`; samples
  without a value are grey "unknown" (before: dropped or white).
* `network.py`: `--diameter` default 0.05 (before: 0.001, too small to see).

Input checks:

* Metadata values are read as text, so numeric sample names (`101`, `007`)
  match the VCF. The scripts stop on an empty or repeated strain name or
  when no VCF sample is in the metadata.
* A VCF that mixes haploid and diploid calls is refused when the haploid
  calls are the majority, else they are counted as missing with a warning
  (before, every haploid call was silently treated as missing).
* `ld_decay.py` and `recombination.py` stop on records not sorted by
  position; `selection.py` sorts them.
* `diversity.py`, `structure.py` and `network.py` warn about samples called
  at fewer than half of the sites and stop if two samples share no called
  site.
* `selection.py` stops with a clear message when no SNP matches the FASTA or
  no gene lies in the VCF region, and on duplicate tip names in the tree.
* Numeric options are checked (for example `--jobs 0` used to hang).

Other changes:

* `generate_citations.py` prints its report; `--out FILE` writes it to a
  file (before: always overwrote `results/citations.txt` in the current
  directory).
* `selection.py --jobs N` sets the number of worker processes.
* Logs record the PopGen version, the command, the working directory, all
  warnings and any error; tables are written under a temporary name and
  renamed when complete.
* `tools/install_phipack.py` does not replace a `Phi` it did not build
  unless given `--force`, and tries the next download URL after an
  interrupted download.
* The code is a Python package (`popgen/`); the `scripts/` commands are
  unchanged, and `pip install -e . --no-deps` optionally adds `popgen-*`
  commands.

## 1.01 (2026-10-22)

* `requirements.txt` pins pyparsing 3.2.3; newer versions emit deprecation
  warnings through matplotlib 3.10.3 and may later stop working with it.
* `tools/install_phipack.py` downloads PhiPack from the bioconda mirror first
  and from the authors' site only if that fails (the authors' site refuses
  many automated downloads).

## 1.00 (2026-10-22)

First public release.

* Analyses: pairwise divergence (`diversity.py`), PCA, DAPC and UPGMA tree
  (`structure.py`), genotype minimum spanning tree (`network.py`), LD decay
  (`ld_decay.py`), Nei–Gojobori dN/dS against the reference (`selection.py`),
  PhiPack recombination tests (`recombination.py`) and a software version
  report (`generate_citations.py`).
* Installation into a local Python virtual environment with pinned package
  versions; `tools/install_phipack.py` builds PhiPack from verified source.
* Simulated example dataset and a pytest suite, including end-to-end runs of
  every script on the example data.
