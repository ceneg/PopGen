Outputs of the example pipeline (`examples/data`), compared byte for byte by `tests/test_golden.py`; generated from the unmodified v1.01 code, then regenerated for the changes named in the refactor commits (B14: `sel.tsv`, `sel.summary.tsv`; B9: `ex.upgma.nwk`; B5: `ld.ld_decay.tsv`).
Regenerate: `POPGEN_UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py tests/test_network.py`, then inspect `git diff tests/golden/`.
Change a golden file only when the change is intended and named in the commit message; any other difference is a regression.
