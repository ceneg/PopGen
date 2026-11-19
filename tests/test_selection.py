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


def test_build_minus_strand_two_exons_snp_in_3prime_exon(sel):
    # Genomic: CC [ATGCTT] NNNN [GACCATAC] CC, gene on the minus strand.
    # The 5' CDS is the right-most exon; its phase 2 drops its last two bases
    # (genomic "AC"), leaving ATGCTT + GACCAT = ATGCTTGACCAT.
    # Reverse complement: ATG GTC AAG CAT (Met Val Lys His).
    ref = "CCATGCTTNNNNGACCATACCC"
    exons = [cds(2, 8, "-", phase=0), cds(12, 20, "-", phase=2)]
    # SNP G->A at genomic position 4 (3' exon), sample 0 only:
    # ATACTT + GACCAT -> revcomp ATG GTC AAG TAT (codon 4 CAT -> TAT)
    v = variants([4], [[1, 0]], [["G", "A"]])
    r, s = sel.build_gene_sequences(ref, exons, v, 2)
    assert r == "ATGGTCAAGCAT"
    assert s == ["ATGGTCAAGTAT", "ATGGTCAAGCAT"]


def test_parse_gff_grouping_and_prefixes(sel, tmp_path):
    gff = tmp_path / "a.gff"
    gff.write_text(
        "##gff-version 3\n"
        "c1\tx\tCDS\t50\t60\t.\t-\t.\tID=cds-a2;Parent=gene-g1\n"
        "c1\tx\tCDS\t10\t20\t.\t-\t1\tID=cds-a1;Parent=gene-g1\n"
        "c1\tx\tCDS\t70\t75\t.\t+\t0\tID=cds-g2\n"
        "c1\tx\tCDS\t80\n"  # fewer than 9 columns
        "##FASTA\n>c1\nACGTACGT\n"
    )
    genes = sel.parse_gff(str(gff))
    assert genes == {
        "g1": [cds(9, 20, "-", phase=1), cds(49, 60, "-", phase=0)],
        "g2": [cds(69, 75, "+", phase=0)],
    }


@pytest.mark.parametrize(
    "value, text",
    [(0, "0.00"), (1e-6, "< 1e-5"), (0.01234, "0.012"), (3.14159, "3.1"),
     (12.34, "12.3"), (float("nan"), "nan"), (float("inf"), "inf")],
)
def test_format_value(sel, value, text):
    assert sel.format_value(value) == text


def one_snp_genes(n_long, n_short, rng):
    """Genes with one synonymous-or-not SNP each on one contig; long ones first."""
    sense = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"
             if a + b + c not in ("TAA", "TAG", "TGA")]
    seq, genes, pos, alleles = [], {}, [], []
    start = 0
    for i in range(n_long + n_short):
        n_codons = 3000 if i < n_long else 30
        cds_seq = "".join(rng.choice(sense, n_codons))
        genes[f"g{i:02d}"] = [cds(start, start + len(cds_seq))]
        ref_base = cds_seq[2]
        alt = "A" if ref_base != "A" else "C"
        pos.append(start + 2)
        alleles.append([ref_base, alt])
        seq.append(cds_seq)
        start += len(cds_seq)
    v = variants(pos, [[1, 0]] * len(pos), alleles)
    return {"c1": "".join(seq)}, {"c1": v}, genes


@pytest.mark.xfail(strict=True, reason="B3")
# The pool forks inside pytest, whose process already runs threads (BLAS);
# Python 3.12+ warns about that. The scripts fork the same way (ADR-4).
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_gene_rows_follow_gff_order(sel):
    # Three chunks of 20 genes; the first chunk is ~100x slower, so with an
    # unordered map its rows come last.
    ref, v, genes = one_snp_genes(20, 25, np.random.default_rng(0))
    for _ in range(2):
        df = sel.calculate_dnds_for_genomes(ref, (v, np.array(["S1", "S2"])), genes)
        assert list(dict.fromkeys(df["gene"])) == list(genes)


@pytest.mark.xfail(strict=True, reason="B4")
def test_contig_name_mismatch_exits(tmp_path):
    import os
    import subprocess
    import sys

    from conftest import PROJECT_ROOT

    data = PROJECT_ROOT / "examples" / "data"
    gff = tmp_path / "chr.gff"
    lines = []
    for line in (data / "example.gff").read_text().splitlines():
        if line and not line.startswith("#"):
            line = line.replace("ctg", "chr", 1)
        lines.append(line)
    gff.write_text("\n".join(lines) + "\n")
    samples = (data / "metadata.tsv").read_text().split()[4::4]
    nwk = tmp_path / "t.nwk"
    nwk.write_text("(" + ",".join(samples) + ");")
    env = dict(os.environ, POPGEN_LOG_DIR=str(tmp_path / "logs"))
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "selection.py"),
         "--vcf", data / "example.vcf", "--gff", gff, "--fasta",
         data / "example.fna", "--nwk", nwk, "--out", tmp_path / "sel"],
        capture_output=True, text=True, env=env,
    )
    out = result.stdout + result.stderr
    assert result.returncode != 0
    assert "chr1" in out and "ctg1" in out
    tsv = tmp_path / "results" / "sel.tsv"
    assert not tsv.exists() or tsv.stat().st_size > 1
