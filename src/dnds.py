"""Nei-Gojobori (1986) synonymous / non-synonymous site and difference counts."""

import itertools
import warnings

import numpy as np
from Bio.Data import CodonTable

VALID_BASES = frozenset("ACGT")


def get_codon_table(genetic_code=1):
    """
    Get the codon table for a given genetic code.
    Defaults to Standard (1).
    """
    return CodonTable.unambiguous_dna_by_id[genetic_code]


def count_site_types(codon, table=None):
    """
    Count synonymous (S) and non-synonymous (N) sites for a given codon.
    Follows Nei-Gojobori (1986).
    """
    if table is None:
        table = get_codon_table()

    s_sites = 0.0
    n_sites = 0.0

    bases = ["T", "C", "A", "G"]

    # Translate original
    try:
        aa_orig = table.forward_table.get(codon)
        if aa_orig is None:  # Stop codon or invalid
            # If codon is stop codon, usually ignored in simple NG86
            # or treated specially.
            # We return 0,0 for stop codons to avoid skewing counts?
            return 0.0, 0.0
    except Exception:
        return 0.0, 0.0

    # For each position
    for pos in range(3):
        orig_base = codon[pos]
        changes = 0
        syn_changes = 0

        for base in bases:
            if base == orig_base:
                continue

            # Mutate
            mut_codon_list = list(codon)
            mut_codon_list[pos] = base
            mut_codon = "".join(mut_codon_list)

            if mut_codon in table.stop_codons:
                continue  # Ignore stop codon mutations pathway

            mut_aa = table.forward_table.get(mut_codon)
            changes += 1
            if mut_aa == aa_orig:
                syn_changes += 1

        if changes > 0:
            fract_syn = syn_changes / changes
            s_sites += fract_syn
            n_sites += 1 - fract_syn

    return s_sites, n_sites


def count_differences(codon1, codon2, table):
    """
    Count synonymous (Sd) and non-synonymous (Nd) differences
    between two codons.
    """
    if codon1 == codon2:
        return 0.0, 0.0

    # Check stop codons
    if codon1 in table.stop_codons or codon2 in table.stop_codons:
        return 0.0, 0.0

    # Unknown codons (lowercase, gaps, IUPAC codes) would translate to None
    # on both sides and be miscounted as synonymous.
    if codon1 not in table.forward_table or codon2 not in table.forward_table:
        return 0.0, 0.0

    aa1 = table.forward_table.get(codon1)
    aa2 = table.forward_table.get(codon2)

    diff_pos = [i for i in range(3) if codon1[i] != codon2[i]]
    num_diffs = len(diff_pos)

    sd, nd = 0.0, 0.0

    if num_diffs == 1:
        if aa1 == aa2:
            sd = 1.0
        else:
            nd = 1.0
    elif num_diffs == 2:
        # Two paths. Average them.
        paths = []
        # Path 1: Change pos1 first, then pos2
        mid1_list = list(codon1)
        mid1_list[diff_pos[0]] = codon2[diff_pos[0]]
        mid1 = "".join(mid1_list)

        # Path 2: Change pos2 first, then pos1
        mid2_list = list(codon1)
        mid2_list[diff_pos[1]] = codon2[diff_pos[1]]
        mid2 = "".join(mid2_list)

        for mid in [mid1, mid2]:
            if mid in table.stop_codons:
                continue  # Invalid path

            # Step 1
            s1, n1 = count_differences(codon1, mid, table)
            # Step 2
            s2, n2 = count_differences(mid, codon2, table)
            paths.append((s1 + s2, n1 + n2))

        if paths:
            avg_s = sum(p[0] for p in paths) / len(paths)
            avg_n = sum(p[1] for p in paths) / len(paths)
            sd, nd = avg_s, avg_n

    elif num_diffs == 3:
        # 6 paths.
        paths = []
        for p in itertools.permutations(diff_pos):
            # p is order of changes, e.g. (0, 1, 2)
            current = list(codon1)
            step_sd, step_nd = 0.0, 0.0
            valid = True

            for idx in p:
                next_codon_list = list(current)
                next_codon_list[idx] = codon2[idx]
                next_codon = "".join(next_codon_list)

                if next_codon in table.stop_codons:
                    valid = False
                    break

                s, n = count_differences("".join(current), next_codon, table)
                step_sd += s
                step_nd += n
                current = next_codon_list

            if valid:
                paths.append((step_sd, step_nd))

        if paths:
            avg_s = sum(p[0] for p in paths) / len(paths)
            avg_n = sum(p[1] for p in paths) / len(paths)
            sd, nd = avg_s, avg_n

    return sd, nd


def calculate_pairwise_dnds(seq1, seq2, genetic_code=1):
    """
    Calculate dN/dS for two aligned coding sequences.

    Sequences are compared case-insensitively. Codons containing anything
    other than A/C/G/T (N, gaps, IUPAC codes) are skipped in both sequences.

    Returns:
        tuple: (pN, pS, N_diff, S_diff, N_sites, S_sites), where pN and pS
        are uncorrected p-distances (N_diff / N_sites, S_diff / S_sites).
    """
    table = get_codon_table(genetic_code)
    seq1 = seq1.upper()
    seq2 = seq2.upper()

    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")

    if len(seq1) % 3 != 0:
        warnings.warn("Sequence length not multiple of 3. Truncating.")
        length = (len(seq1) // 3) * 3
        seq1 = seq1[:length]
        seq2 = seq2[:length]

    S_sites = 0.0
    N_sites = 0.0
    S_diff = 0.0
    N_diff = 0.0

    # Iterate codons
    for i in range(0, len(seq1), 3):
        codon1 = seq1[i : i + 3]
        codon2 = seq2[i : i + 3]

        if len(codon1) != 3 or len(codon2) != 3:
            continue
        if not (VALID_BASES.issuperset(codon1) and VALID_BASES.issuperset(codon2)):
            continue

        # Count Sites (Average of both)
        s1, n1 = count_site_types(codon1, table)
        s2, n2 = count_site_types(codon2, table)
        S_sites += (s1 + s2) / 2
        N_sites += (n1 + n2) / 2

        # Count Differences
        sd, nd = count_differences(codon1, codon2, table)
        S_diff += sd
        N_diff += nd

    # Uncorrected p-distances; see jukes_cantor() for dN, dS
    pN = N_diff / N_sites if N_sites > 0 else 0.0
    pS = S_diff / S_sites if S_sites > 0 else 0.0

    return pN, pS, N_diff, S_diff, N_sites, S_sites


def jukes_cantor(p):
    """
    Jukes-Cantor (1969) correction of a p-distance, as used by Nei-Gojobori
    to turn pN, pS into dN, dS: d = -3/4 ln(1 - 4p/3).

    Args:
        p: Proportion of differing sites (scalar or array).

    Returns:
        Corrected distance; NaN where p >= 3/4 (saturated) or p is NaN.
    """
    p = np.asarray(p, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = -0.75 * np.log1p(-4.0 * p / 3.0)
    d = np.where(p < 0.75, d, np.nan)
    return float(d) if d.ndim == 0 else d
