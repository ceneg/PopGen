import numpy as np
import pandas as pd
import pytest

VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
)


@pytest.fixture(scope="module")
def sel(load_script):
    return load_script("selection")


def cds(start, end, strand="+", phase=0, chrom="c1"):
    return {
        "chrom": chrom, "start": start, "end": end, "strand": strand, "phase": phase
    }


def variants(pos, gt, alleles):
    return {"pos": np.array(pos), "gt": np.array(gt), "alleles": np.array(alleles)}


def test_build_plus_strand_two_exons(sel):
    ref = "AAATGGCCxxxxTTTAAA"
    # exon1 = ATGGCC (2..8), exon2 = TTT (12..15)
    v = variants([3, 13], [[1, 0], [0, -1]], [["T", "C"], ["T", "G"]])
    r, s = sel.build_gene_sequences(ref, [cds(2, 8), cds(12, 15)], v, 2)
    assert r == "ATGGCCTTT"
    assert s[0] == "ACGGCCTTT"
    assert s[1] == "ATGGCCTNT"  # missing call -> N


def test_build_minus_strand_and_phase(sel):
    ref = "GCCATGA"
    # minus strand, 5' CDS is the right-most; phase 1 trims its last base
    r, s = sel.build_gene_sequences(ref, [cds(0, 7, "-", phase=1)], None, 1)
    assert r == "CATGGC"  # revcomp of GCCATG
    assert s == [r]


def test_build_uses_called_alt_allele(sel):
    v = variants([0], [[2]], [["A", "C", "G"]])
    r, s = sel.build_gene_sequences("AAA", [cds(0, 3)], v, 1)
    assert s[0] == "GAA"


def test_parse_gff_phase(sel, tmp_path):
    gff = tmp_path / "a.gff"
    gff.write_text(
        "c1\tx\tCDS\t1\t6\t.\t+\t2\tID=cds-1;Parent=rna-1\n"
        "c1\tx\tgene\t1\t6\t.\t+\t.\tID=gene-1\n"
    )
    genes = sel.parse_gff(str(gff))
    assert genes == {"rna-1": [cds(0, 6, "+", phase=2)]}


def test_reconstruct_skips_indels_and_ref_mismatch(sel, tmp_path):
    vcf = tmp_path / "v.vcf"
    vcf.write_text(
        VCF_HEADER
        + "c1\t2\t.\tA\tG\t.\tPASS\t.\tGT\t1\t0\n"  # ok
        + "c1\t3\t.\tA\tAT\t.\tPASS\t.\tGT\t1\t0\n"  # indel
        + "c1\t4\t.\tT\tC\t.\tPASS\t.\tGT\t1\t0\n"  # REF mismatch (ref has A)
    )
    v, samples = sel.reconstruct_genomes_from_vcf(str(vcf), {"c1": "AAAAAA"})
    assert list(samples) == ["S1", "S2"]
    np.testing.assert_array_equal(v["c1"]["pos"], [1])


def test_summary_pools_genes_without_differences(sel):
    df = pd.DataFrame(
        {
            "gene": ["g1", "g2"],
            "strain": ["S1", "S1"],
            "N_diff": [1.0, 0.0],
            "S_diff": [1.0, 0.0],
            "N_sites": [30.0, 70.0],
            "S_sites": [10.0, 10.0],
        }
    )
    out = sel.summarise_by_strain(df)
    assert out.loc["S1", "pN"] == pytest.approx(0.01)
    assert out.loc["S1", "pS"] == pytest.approx(0.05)
    d_n = -0.75 * np.log(1 - 4 * 0.01 / 3)
    d_s = -0.75 * np.log(1 - 4 * 0.05 / 3)
    assert out.loc["S1", "dN"] == pytest.approx(d_n)
    assert out.loc["S1", "dS"] == pytest.approx(d_s)
    assert out.loc["S1", "dN/dS"] == pytest.approx(d_n / d_s)
    assert out.loc["S1", "genes"] == 2


def test_genes_in_vcf_region(sel):
    genes = {
        "inside": [{"chrom": "c1", "start": 100, "end": 200}],
        "overlap": [{"chrom": "c1", "start": 450, "end": 600}],
        "after": [{"chrom": "c1", "start": 600, "end": 700}],
        "other_contig": [{"chrom": "c2", "start": 0, "end": 50}],
    }
    variants = {"c1": {"pos": np.array([150, 500])}}
    assert set(sel.genes_in_vcf_region(genes, variants)) == {"inside", "overlap"}
