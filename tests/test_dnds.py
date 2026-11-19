import math

import numpy as np
import pytest

from popgen.dnds import (
    jukes_cantor,
    calculate_pairwise_dnds,
    count_differences,
    count_site_types,
    dnds_ratio,
    get_codon_table,
)


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


SENSE = [a + b + c for a in "TCAG" for b in "TCAG" for c in "TCAG"
         if a + b + c not in ("TAA", "TAG", "TGA")]


def test_site_counts_of_sense_codons():
    table = get_codon_table(1)
    assert len(SENSE) == 61
    for codon in SENSE:
        s, n = count_site_types(codon, table)
        assert 0 < s + n <= 3 + 1e-12


def test_count_differences_symmetric():
    table = get_codon_table(1)
    rng = np.random.default_rng(7)
    for _ in range(200):
        a, b = rng.choice(SENSE, 2)
        assert count_differences(a, b, table) == count_differences(b, a, table)


def test_count_differences_three_positions():
    # TTT (Phe) -> GGG (Gly), no path passes through a stop. Per path (S, N):
    # 012: GTT GGT GGG = (1, 2); 021: GTT GTG GGG = (1, 2);
    # 102: TGT GGT GGG = (1, 2); 120: TGT TGG GGG = (0, 3);
    # 201: TTG GTG GGG = (0, 3); 210: TTG TGG GGG = (0, 3).
    # Mean over 6 paths: S = 3/6, N = 15/6.
    table = get_codon_table(1)
    sd, nd = count_differences("TTT", "GGG", table)
    assert sd == pytest.approx(0.5)
    assert nd == pytest.approx(2.5)


def test_stop_codon_pair_is_skipped():
    """T-DNDS-1 (B14): Trp -> stop contributes no sites and no differences."""
    pn, ps, nd, sd, n_sites, s_sites = calculate_pairwise_dnds("TGGCCT", "TGACCC")
    assert n_sites + s_sites == pytest.approx(3)  # only CCT / CCC
    assert sd == 1 and nd == 0


def test_only_stop_pathways_give_none():
    """B14: no valid pathway -> None (the caller skips the pair)."""
    from types import SimpleNamespace

    codons = [a + b + c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
    stops = ["ACA", "AAC"]  # both intermediates of AAA -> ACC
    table = SimpleNamespace(
        stop_codons=stops,
        forward_table={c: "K" if c == "AAA" else "T" for c in codons if c not in stops},
    )
    assert count_differences("AAA", "ACC", table) is None
    # The standard code has no sense pair whose every pathway hits a stop
    std = get_codon_table()
    sense = [c for c in codons if c in std.forward_table]
    assert all(count_differences(x, y, std) is not None for x in sense for y in sense)


@pytest.mark.parametrize("dn, ds, expected", [
    (0.1, 0.0, math.inf),
    (0.0, 0.0, math.nan),
    (math.nan, 0.1, math.nan),
    (0.2, 0.1, 2.0),
])
def test_dnds_ratio(dn, ds, expected):
    """T-DNDS-2 (B14)."""
    got = dnds_ratio(dn, ds)
    assert isinstance(got, float)
    if math.isnan(expected):
        assert math.isnan(got)
    else:
        assert got == pytest.approx(expected)


def test_dnds_ratio_array():
    got = dnds_ratio(np.array([0.1, 0.0, 0.2]), np.array([0.0, 0.0, 0.1]))
    assert isinstance(got, np.ndarray)
    np.testing.assert_array_equal(got, [np.inf, np.nan, 2.0])
