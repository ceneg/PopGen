import pytest
import os
import sys

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math

import numpy as np

from src.dnds import (
    jukes_cantor,
    calculate_pairwise_dnds,
    count_differences,
    count_site_types,
    get_codon_table,
)
from src.validators import validate_metadata


# Mock Data
def test_dnds_identical():
    seq = "ATGCGT"  # Met-Arg
    dN, dS, Nd, Ns, N_site, S_site = calculate_pairwise_dnds(seq, seq)
    assert dN == 0
    assert dS == 0
    assert Nd == 0
    assert Ns == 0
    assert N_site > 0
    assert S_site > 0


def test_dnds_synonymous():
    # TTT (Phe) -> TTC (Phe)
    seq1 = "TTT"
    seq2 = "TTC"
    dN, dS, Nd, Ns, N_site, S_site = calculate_pairwise_dnds(seq1, seq2)
    assert Nd == 0
    assert Ns == 1.0  # 1 Syn difference
    assert dN == 0
    assert dS > 0


def test_dnds_nonsynonymous():
    # TTT (Phe) -> TTG (Leu)
    seq1 = "TTT"
    seq2 = "TTG"
    dN, dS, Nd, Ns, N_site, S_site = calculate_pairwise_dnds(seq1, seq2)
    assert Nd == 1.0
    assert Ns == 0
    assert dN > 0
    assert dS == 0


def test_ng86_counts():
    table = get_codon_table(1)
    # Proline CCN. CC T/C/A/G are all Pro.
    # So 3rd position is 4-fold degenerate -> 1 synonymous site.
    s, n = count_site_types("CCT", table)
    # Position 1: C->T (TCT=Ser, N), C->A (ACT=Thr, N), C->G (GCT=Ala, N). 0 syn.
    # Position 2: C->T (CTT=Leu, N), C->A (CAT=His, N), C->G (CGT=Arg, N). 0 syn.
    # Position 3: T->C (CCC=Pro, S), T->A (CCA=Pro, S), T->G (CCG=Pro, S). 3 syn.
    # Total S changes = 3. Total changes = 9.
    # fraction S = 1/3?
    # NG86 says: S_site = sum(fi).
    # Pos 1: f=0. Pos 2: f=0. Pos 3: f=3/3=1.
    # Total S_site = 1.
    assert abs(s - 1.0) < 0.01


def test_validator_metadata(tmp_path):
    f = tmp_path / "meta.tsv"
    f.write_text("strain\tgroup\nA\t1\nB\t2")
    assert validate_metadata(str(f))


def test_validator_bad_metadata(tmp_path):
    f = tmp_path / "bad.tsv"
    f.write_text("id\tgroup\nA\t1")  # Wrong column
    with pytest.raises(ValueError, match="strain"):
        validate_metadata(str(f))


def test_dnds_lowercase_matches_uppercase():
    # Soft-masked reference (lowercase) must give the same counts
    lower = calculate_pairwise_dnds("ttt", "TTC")
    assert lower == calculate_pairwise_dnds("TTT", "TTC")


@pytest.mark.parametrize("codon", ["T-T", "TRT", "NNN", "TT?"])
def test_dnds_ambiguous_codons_skipped(codon):
    dN, dS, Nd, Sd, N_site, S_site = calculate_pairwise_dnds("TTTCCT", codon + "CCC")
    # Only the second codon (CCT -> CCC, synonymous) is compared
    assert Nd == 0
    assert Sd == 1.0
    assert abs(S_site - 1.0) < 1e-9


def test_count_differences_unknown_codon_not_synonymous():
    table = get_codon_table(1)
    assert count_differences("ttt", "TTC", table) == (0.0, 0.0)


def test_count_differences_two_positions():
    table = get_codon_table(1)
    # TTT (Phe) -> TCC (Ser): paths via TCT (Ser) [N then S] and TTC (Phe) [S then N]
    sd, nd = count_differences("TTT", "TCC", table)
    assert sd == 1.0
    assert nd == 1.0


def test_jukes_cantor():
    assert jukes_cantor(0.0) == 0.0
    assert jukes_cantor(0.1) == pytest.approx(-0.75 * math.log(1 - 0.4 / 3))
    assert math.isnan(jukes_cantor(0.75))
    np.testing.assert_allclose(jukes_cantor(np.array([0.0, 0.2])),
                               [0.0, -0.75 * math.log(1 - 0.8 / 3)])
