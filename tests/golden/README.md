Outputs of the example pipeline (`examples/data`), compared byte for byte by `tests/test_golden.py`; generated from the unmodified v1.01 code.
Regenerate: `POPGEN_UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py tests/test_network.py`, then inspect `git diff tests/golden/`.
Change a golden file only when the change is intended and named in the commit message; any other difference is a regression.
