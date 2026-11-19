#!/usr/bin/env python3
"""Run `popgen.analyses.ld_decay` from a checkout without installation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from popgen.analyses.ld_decay import main

if __name__ == "__main__":
    main()
