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


# The pool forks inside pytest, whose process already runs threads (BLAS);
# Python 3.12+ warns about that. The scripts fork the same way (ADR-4).
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_gene_rows_follow_gff_order(sel):
    """T-SEL-1 (B3)."""
    # Three chunks of 20 genes; the first chunk is ~100x slower, so with an
    # unordered map its rows would come last.
    ref, v, genes = one_snp_genes(20, 25, np.random.default_rng(0))
    for _ in range(2):
        df, errors = sel.run_genes(ref, v, np.array(["S1", "S2"]), genes, 3)
        assert not errors
        assert list(dict.fromkeys(df["gene"])) == list(genes)
        assert list(df["strain"][:2]) == ["S1", "S2"]


def test_contig_name_mismatch_exits(tmp_path):
    """T-SEL-2 (B4)."""
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


def run_selection(tmp_path, vcf, gff=None):
    from conftest import PROJECT_ROOT, run_script

    data = PROJECT_ROOT / "examples" / "data"
    samples = (data / "metadata.tsv").read_text().split()[4::4]
    nwk = tmp_path / "t.nwk"
    nwk.write_text("(" + ",".join(f"{s}:1" for s in samples) + ");")
    return run_script("selection", tmp_path, "--vcf", vcf, "--gff",
                      gff or data / "example.gff", "--fasta", data / "example.fna",
                      "--nwk", nwk, "--out", tmp_path / "sel", "--jobs", "2")


def test_no_difference_writes_header_and_zero_summary(tmp_path):
    """T-SEL-3 (B4): every call is reference."""
    from conftest import PROJECT_ROOT

    lines = []
    src = PROJECT_ROOT / "examples" / "data" / "example.vcf"
    for line in src.read_text().splitlines():
        if not line.startswith("#"):
            f = line.split("\t")
            line = "\t".join(f[:9] + ["0"] * (len(f) - 9))
        lines.append(line)
    vcf = tmp_path / "ref.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    result = run_selection(tmp_path, vcf)
    assert result.returncode == 0, result.stdout + result.stderr
    res = tmp_path / "results"
    tsv = pd.read_csv(res / "sel.tsv", sep="\t")
    assert tsv.empty and list(tsv.columns) == sel_columns()
    summary = pd.read_csv(res / "sel.summary.tsv", sep="\t", index_col=0)
    assert len(summary) == 24
    assert (summary[["N_diff", "S_diff", "dN", "dS"]] == 0).all().all()


def sel_columns():
    return ["gene", "strain", "dN", "dS", "dN/dS", "pN", "pS",
            "N_diff", "S_diff", "N_sites", "S_sites"]


def test_fasta_name_mismatch_exits(tmp_path):
    """B4: no VCF contig in the FASTA."""
    from conftest import PROJECT_ROOT

    src = PROJECT_ROOT / "examples" / "data" / "example.vcf"
    vcf = tmp_path / "chr.vcf"
    vcf.write_text("\n".join(
        line if line.startswith("#") else line.replace("ctg", "chr", 1)
        for line in src.read_text().splitlines()) + "\n")
    result = run_selection(tmp_path, vcf)
    out = result.stdout + result.stderr
    assert result.returncode != 0
    assert "chr1" in out and "ctg1" in out
    assert not (tmp_path / "results" / "sel.tsv").exists()


def test_report_failures(sel, caplog):
    errors = [f"g{i}: boom" for i in range(25)]
    with caplog.at_level("WARNING"):
        sel.report_failures(errors, 30)
    text = caplog.text
    assert "25 of 30 genes failed" in text and "g19: boom" in text
    assert "g20: boom" not in text and "and 5 more" in text
    with pytest.raises(SystemExit):
        sel.report_failures(errors, 25)


def test_count_partial_codons(sel):
    genes = {"ok": [cds(0, 9, phase=0)], "bad": [cds(0, 10, phase=0)],
             "phase": [cds(0, 10, phase=1)]}
    assert sel.count_partial_codons(genes) == 1


@pytest.fixture
def two_sample_vcf(tmp_path):
    vcf = tmp_path / "s.vcf"
    vcf.write_text(VCF_HEADER + "c1\t5\t.\tA\tC\t.\tPASS\t.\tGT\t1\t0\n")
    return vcf


@pytest.mark.parametrize("newick", [None, "(X:1,Y:1);", "(S1:1,S1:2,S2:1);"])
def test_validate_tree_exits(sel, tmp_path, two_sample_vcf, newick):
    """T-SEL-6 (B12): missing file, zero overlap, duplicate tips."""
    nwk = tmp_path / "t.nwk"
    if newick is not None:
        nwk.write_text(newick)
    with pytest.raises(SystemExit):
        sel.validate_tree(str(nwk), str(two_sample_vcf))


def test_validate_tree_partial_overlap_warns(sel, tmp_path, two_sample_vcf, caplog):
    """T-SEL-6 (B12): one warning per direction; no branch lengths warns."""
    nwk = tmp_path / "t.nwk"
    nwk.write_text("(S1:1,X:1);")
    with caplog.at_level("WARNING"):
        tree = sel.validate_tree(str(nwk), str(two_sample_vcf))
    assert tree.count_terminals() == 2
    assert "1 samples in VCF are missing" in caplog.text
    assert "1 samples in Tree are missing" in caplog.text
    assert "no branch lengths" not in caplog.text
    nwk.write_text("(S1,S2);")
    with caplog.at_level("WARNING"):
        sel.validate_tree(str(nwk), str(two_sample_vcf))
    assert "no branch lengths" in caplog.text


def test_draw_annotated_tree_keeps_topology(sel, tmp_path, monkeypatch):
    """T-SEL-7 (B12): the tree is drawn as given, not re-clustered."""
    import io

    import matplotlib.pyplot as plt
    from Bio import Phylo
    from popgen.io import ensure_output_dirs

    tree = Phylo.read(io.StringIO("((a:1,b:3):1,c:5);"), "newick")
    paths = ensure_output_dirs(tmp_path / "t")
    figs = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", figs.append)  # keep the figure to inspect
    sel.draw_annotated_tree(tree, {"a": 0.5, "b": 0.012}, "dN", "dN", paths)
    monkeypatch.undo()
    assert paths.png("dN.tree").exists() and paths.svg("dN.tree").exists()
    ax = figs[0].axes[0]
    tips = sorted((t for t in ax.texts if "(dN:" in t.get_text()),
                  key=lambda t: t.get_position()[1])
    assert [t.get_text().strip() for t in tips] == [
        "a (dN: 0.50)", "b (dN: 0.012)", "c (dN: N/A)"]
    real_close(figs[0])


@pytest.mark.parametrize("cds_list", [
    [cds(5, 15, "-")],                       # runs past the contig end
    [cds(0, 3), cds(4, 7, chrom="c2")],      # one gene on two contigs
    [cds(0, 3), cds(4, 7, "-")],             # one gene on two strands
])
def test_build_rejects_inconsistent_cds(sel, cds_list):
    with pytest.raises(ValueError):
        sel.build_gene_sequences("ACGTACGTACGTA", cds_list, None, 1)
