# Changelog

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
