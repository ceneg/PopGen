"""
dN/dS of every sample against the reference (Nei-Gojobori 1986, Jukes-Cantor
corrected), per gene and pooled genome-wide, with the genome-wide values drawn
on a tree.
"""

import argparse
import logging
import math
import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO
from Bio.Seq import Seq

from popgen.dnds import calculate_pairwise_dnds, dnds_ratio, jukes_cantor
from popgen.cli import add_out_argument, positive_int
from popgen.genotypes import contig_rows, haploid_alleles
from popgen.io import ensure_output_dirs, read_vcf, safe_name, write_table
from popgen.plotting import save_figure
from popgen.runlog import die, setup_logging

log = logging.getLogger("selection")

# Genes per task of the worker pool
GENE_CHUNK = 20
# Columns of the per-gene table
RESULT_COLUMNS = ["gene", "strain", "dN", "dS", "dN/dS", "pN", "pS",
                  "N_diff", "S_diff", "N_sites", "S_sites"]
# Failed genes listed one by one in the log
MAX_LISTED_FAILURES = 20


def first_names(names, n: int = 3) -> str:
    """The first ``n`` names, comma-separated, with "..." if there are more."""
    names = [str(x) for x in names]
    return ", ".join(names[:n]) + (", ..." if len(names) > n else "")


def report_failures(errors: list[str], n_genes: int) -> None:
    """Log every failed gene (at most ``MAX_LISTED_FAILURES``); exit if all failed."""
    if not errors:
        return
    log.warning(f"Warning: {len(errors)} of {n_genes} genes failed and were skipped:")
    for err in errors[:MAX_LISTED_FAILURES]:
        log.warning(f"  {err}")
    if len(errors) > MAX_LISTED_FAILURES:
        log.warning(f"  and {len(errors) - MAX_LISTED_FAILURES} more")
    if len(errors) == n_genes:
        die("every gene failed; see the messages above.")


def count_partial_codons(genes: dict) -> int:
    """Genes whose CDS length after the phase trim is not a multiple of 3."""
    n = 0
    for cds_list in genes.values():
        minus = cds_list[0]["strand"] == "-"
        phase = cds_list[-1]["phase"] if minus else cds_list[0]["phase"]
        n += (sum(c["end"] - c["start"] for c in cds_list) - phase) % 3 != 0
    return n


def load_reference_genome(fasta_path: str) -> dict[str, str]:
    """
    Load the reference as {contig: uppercase sequence string}. Uppercasing
    removes soft-masking, whose lowercase codons are not in the codon table.
    """
    log.info(f"Loading reference genome from: {fasta_path}")
    try:
        ref_genome = {
            rec.id: str(rec.seq).upper() for rec in SeqIO.parse(fasta_path, "fasta")
        }
    except Exception as e:
        die(f"could not load the reference genome: {e}")
    if not ref_genome:
        die(f"no sequences found in {fasta_path}")
    log.info(f"Loaded {len(ref_genome)} contigs.")
    return ref_genome


def parse_gff(gff_path: str) -> dict[str, list[dict]]:
    """
    Collect CDS features per gene (transcript) from a GFF3 file.

    CDS lines are grouped by their ``Parent`` attribute (else ``gene``, else
    ``ID``), without a leading ``gene-`` or ``cds-``.

    Returns:
        dict: {gene_id: [{"chrom", "start", "end", "strand", "phase"}]}, with
        0-based half-open coordinates, sorted by start.
    """
    log.info(f"Parsing GFF file: {gff_path}")
    genes = defaultdict(list)
    try:
        with open(gff_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue

                chrom, _, feature, start, end, _, strand, phase, attr = parts
                if feature != "CDS":
                    continue

                attr_dict = {}
                for item in attr.split(";"):
                    if "=" in item:
                        key, val = item.strip().split("=", 1)
                        attr_dict[key] = val

                gene_id = (
                    attr_dict.get("Parent")
                    or attr_dict.get("gene")
                    or attr_dict.get("ID")
                )
                if gene_id:
                    if gene_id.startswith("gene-"):
                        gene_id = gene_id[5:]
                    if gene_id.startswith("cds-"):
                        gene_id = gene_id[4:]

                if gene_id:
                    genes[gene_id].append(
                        {
                            "chrom": chrom,
                            "start": int(start) - 1,  # 0-based
                            "end": int(end),  # 0-based exclusive (GFF is 1-inclusive)
                            "strand": strand,
                            "phase": int(phase) if phase in ("0", "1", "2") else 0,
                        }
                    )

        for gene in genes:
            genes[gene].sort(key=lambda x: x["start"])

        log.info(f"Found {len(genes)} genes with CDS entries.")
        return genes

    except Exception as e:
        die(f"could not parse the GFF: {e}")


def reconstruct_genomes_from_vcf(vcf_path: str, ref_genome: dict[str, str]) -> tuple:
    """
    Collect the SNPs needed to reconstruct each sample's coding sequences.

    Only single-base REF/ALT variants are used (indels would shift the
    reading frame), and only those whose REF matches the reference FASTA.
    Uses the first allele of each call (haploid data, or homozygous diploid;
    heterozygous calls are counted in a warning).

    Returns: (variants_by_chrom, samples), where variants_by_chrom is
    {chrom: {"pos": 0-based sorted positions, "gt": (n, n_samples) allele
    indices (-1 = missing), "alleles": (n, n_alleles) REF + ALT bases}}.
    """
    log.info("Reconstructing genomes from VCF...")
    samples, gt, callset = read_vcf(
        vcf_path,
        fields=[
            "samples",
            "calldata/GT",
            "variants/POS",
            "variants/CHROM",
            "variants/ALT",
            "variants/REF",
        ],
    )

    gt_hap = haploid_alleles(gt, "first")  # Haploid, or first allele (warned)

    chroms = callset["variants/CHROM"]
    poss = callset["variants/POS"] - 1
    refs = callset["variants/REF"].astype(str)
    alts = callset["variants/ALT"].astype(str)
    if alts.ndim == 1:
        alts = alts[:, None]
    alleles = np.char.upper(np.column_stack([refs, alts]))

    # SNP: one-base REF and every non-empty ALT one base
    lens = np.char.str_len(alleles)
    is_snp = (lens[:, 0] == 1) & np.all((lens[:, 1:] == 1) | (lens[:, 1:] == 0), axis=1)
    if (~is_snp).any():
        log.info(f"Skipping {int((~is_snp).sum())} non-SNP variants (indels / MNPs).")

    variants_by_chrom = {}
    n_ref_mismatch = 0
    n_off_contig = 0
    rows_by_contig = contig_rows(chroms, poss, "sort")
    for chrom, idx in rows_by_contig.items():  # VCF order
        idx = idx[is_snp[idx]]  # sorted by position
        if chrom not in ref_genome:
            n_off_contig += len(idx)
            continue
        seq = ref_genome[chrom]
        in_range = (poss[idx] >= 0) & (poss[idx] < len(seq))
        n_off_contig += int((~in_range).sum())
        idx = idx[in_range]
        ref_base = np.array([seq[p] for p in poss[idx]], dtype="U1")
        matches = ref_base == alleles[idx, 0]
        n_ref_mismatch += int((~matches).sum())
        idx = idx[matches]
        if len(idx):
            variants_by_chrom[chrom] = {
                "pos": poss[idx],
                "gt": gt_hap[idx],
                "alleles": alleles[idx],
            }

    if n_off_contig:
        log.warning(f"Warning: skipped {n_off_contig} SNPs outside the FASTA "
                    "contigs.")
    if n_ref_mismatch:
        log.warning(f"Warning: skipped {n_ref_mismatch} SNPs whose REF mismatches "
                    "the FASTA.")
    if not variants_by_chrom:
        die("no SNP of the VCF could be placed on the reference: check that the "
            f"VCF contigs ({first_names(rows_by_contig)}) match the FASTA record "
            f"ids ({first_names(ref_genome)}) and that REF matches the FASTA.")

    return variants_by_chrom, samples


# Data shared by the worker processes (set by init_worker)
worker_ref_genome = None
worker_variants = None
worker_samples = None


def init_worker(ref_genome, variants_by_chrom, samples) -> None:
    """Set the data shared by all workers (module globals)."""
    global worker_ref_genome, worker_variants, worker_samples
    worker_ref_genome = ref_genome
    worker_variants = variants_by_chrom
    worker_samples = samples


def build_gene_sequences(
    ref_seq: str, cds_list: list[dict], variants: dict | None, n_samples: int
) -> tuple[str, list[str]]:
    """
    Build the reference and per-sample coding sequences of one gene, in
    5'->3' orientation and starting at the first complete codon (GFF phase).

    Args:
        ref_seq: Uppercase contig sequence.
        cds_list: CDS dicts of the gene, sorted by start.
        variants: The contig's entry from ``reconstruct_genomes_from_vcf`` or None.
        n_samples: Number of samples.

    Returns:
        tuple: (reference CDS string, list of sample CDS strings). Missing
        calls become N, which removes that codon from the comparison.
    """
    if len({(c["chrom"], c["strand"]) for c in cds_list}) > 1:
        raise ValueError("CDS features on more than one contig or strand")
    for c in cds_list:
        if not 0 <= c["start"] < c["end"] <= len(ref_seq):
            raise ValueError(f"CDS {c['start'] + 1}..{c['end']} outside contig "
                             f"{c['chrom']} (length {len(ref_seq)})")
    full_ref = "".join(ref_seq[c["start"] : c["end"]] for c in cds_list)
    sample_seqs = [full_ref] * n_samples

    if variants is not None:
        v_pos = variants["pos"]
        s_idx = np.searchsorted(v_pos, cds_list[0]["start"])
        e_idx = np.searchsorted(v_pos, max(c["end"] for c in cds_list))
        if s_idx < e_idx:
            # Map genomic positions of SNPs inside a CDS to positions in full_ref
            offset = 0
            hits, cds_pos = [], []
            for cds in cds_list:
                start, end = cds["start"], cds["end"]
                rel = np.arange(s_idx, e_idx)
                inside = rel[(v_pos[rel] >= start) & (v_pos[rel] < end)]
                hits.extend(inside)
                cds_pos.extend(v_pos[inside] - start + offset)
                offset += end - start

            if hits:
                hits = np.asarray(hits)
                g = variants["gt"][hits]  # (n_hits, n_samples)
                alleles = variants["alleles"][hits]
                for samp_idx in range(n_samples):
                    gi = g[:, samp_idx]
                    changed = gi != 0
                    if not changed.any():
                        continue
                    seq_list = list(full_ref)
                    for k in np.flatnonzero(changed):
                        a = gi[k]
                        if a < 0 or a >= alleles.shape[1] or not alleles[k, a]:
                            seq_list[cds_pos[k]] = "N"
                        else:
                            seq_list[cds_pos[k]] = alleles[k, a]
                    sample_seqs[samp_idx] = "".join(seq_list)

    # Drop the leading partial codon (phase of the 5' CDS), then orient 5'->3'
    minus = cds_list[0]["strand"] == "-"
    phase = cds_list[-1]["phase"] if minus else cds_list[0]["phase"]

    def trim(x):
        if minus:
            return str(Seq(x[: len(x) - phase]).reverse_complement())
        return x[phase:]

    ref_out = trim(full_ref)
    return ref_out, [ref_out if x is full_ref else trim(x) for x in sample_seqs]


def process_gene(item: tuple[str, list[dict]]) -> tuple[list[dict], str | None]:
    """
    Nei-Gojobori counts of one gene for every sample (run in a worker).

    Args:
        item: (gene_id, CDS list).

    Returns:
        tuple: (one result dict per sample, error message or None).
    """
    gene_id, cds_list = item

    chrom = cds_list[0]["chrom"]
    if chrom not in worker_ref_genome:
        return [], None

    try:
        r_seq, s_seqs = build_gene_sequences(
            worker_ref_genome[chrom],
            cds_list,
            worker_variants.get(chrom),
            len(worker_samples),
        )
        results = []
        cache = {}
        for sample, s_seq in zip(worker_samples, s_seqs):
            if s_seq not in cache:
                cache[s_seq] = calculate_pairwise_dnds(r_seq, s_seq)
            pN, pS, n_diff, s_diff, n_sites, s_sites = cache[s_seq]
            dN, dS = jukes_cantor(pN), jukes_cantor(pS)
            results.append(
                {
                    "gene": gene_id,
                    "strain": sample,
                    "dN": dN,
                    "dS": dS,
                    "dN/dS": dnds_ratio(dN, dS),
                    "pN": pN,
                    "pS": pS,
                    "N_diff": n_diff,
                    "S_diff": s_diff,
                    "N_sites": n_sites,
                    "S_sites": s_sites,
                }
            )
        return results, None
    except Exception as e:
        return [], f"{gene_id}: {e}"


def default_jobs() -> int:
    """CPUs available to this process (CPU affinity where known), minus one."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        n = os.cpu_count() or 1
    return max(1, n - 1)


def run_genes(
    ref_genome, variants_by_chrom, samples, genes: dict, jobs: int
) -> tuple[pd.DataFrame, list[str]]:
    """
    Count every gene in ``jobs`` worker processes.

    With the fork start method the workers inherit the (read-only) data
    copy-on-write instead of each receiving a pickled copy (ADR-4); other start
    methods pass it to ``init_worker``.

    Returns:
        tuple: (DataFrame with one row per gene and sample, in gene order and
        then sample order; error messages of the genes that failed).
    """
    log.info(f"Calculating dN/dS using {jobs} parallel processes...")
    items = list(genes.items())

    init_worker(ref_genome, variants_by_chrom, samples)
    if "fork" in mp.get_all_start_methods():
        executor = ProcessPoolExecutor(max_workers=jobs,
                                       mp_context=mp.get_context("fork"))
    else:
        executor = ProcessPoolExecutor(
            max_workers=jobs,
            initializer=init_worker,
            initargs=(ref_genome, variants_by_chrom, samples),
        )

    all_results, errors = [], []
    try:
        with executor:
            results = executor.map(process_gene, items, chunksize=GENE_CHUNK)
            for count, (res, err) in enumerate(results, 1):
                all_results.extend(res)
                if err:
                    errors.append(err)
                if count % 500 == 0:
                    log.info(f"Processed {count}/{len(items)} genes...")
    except BrokenProcessPool:
        die("a worker process died (out of memory?); try a smaller --jobs.")

    log.info(f"Finished processing {len(items)} genes.")
    return pd.DataFrame(all_results, columns=RESULT_COLUMNS), errors


def genes_in_vcf_region(genes: dict, variants_by_chrom: dict) -> dict:
    """
    Keep genes that overlap the region spanned by the VCF's SNPs (first to last
    SNP of each contig). Genes outside it have no genotype information, so
    counting their sites as invariant would deflate genome-wide dN and dS
    (e.g. for a VCF restricted to one region).

    Args:
        genes: {gene_id: CDS list} from ``parse_gff``.
        variants_by_chrom: From ``reconstruct_genomes_from_vcf``.

    Returns:
        dict: The retained genes.
    """
    spans = {c: (v["pos"][0], v["pos"][-1]) for c, v in variants_by_chrom.items()}
    kept = {}
    for gene_id, cds_list in genes.items():
        span = spans.get(cds_list[0]["chrom"])
        if span is None:
            continue
        start = min(c["start"] for c in cds_list)
        end = max(c["end"] for c in cds_list)
        if start <= span[1] and end > span[0]:
            kept[gene_id] = cds_list
    return kept


def summarise_by_strain(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genome-wide dN, dS per strain, pooled over all genes: total differences
    divided by total sites (pN, pS), then Jukes-Cantor corrected. Genes
    without differences count as sites.
    """
    count_cols = ["N_diff", "S_diff", "N_sites", "S_sites"]
    sums = results_df.groupby("strain")[count_cols].sum()
    strain_df = pd.DataFrame(index=sums.index)
    p_n = sums["N_diff"] / sums["N_sites"]
    p_s = sums["S_diff"] / sums["S_sites"]
    strain_df["dN"] = jukes_cantor(p_n.to_numpy())
    strain_df["dS"] = jukes_cantor(p_s.to_numpy())
    strain_df["dN/dS"] = dnds_ratio(strain_df["dN"], strain_df["dS"])
    strain_df["pN"] = p_n
    strain_df["pS"] = p_s
    strain_df = pd.concat([strain_df, sums], axis=1)
    strain_df["genes"] = results_df.groupby("strain").size()
    return strain_df


def format_value(val) -> str:
    """Format a tree label value: two significant digits below 10, else one decimal."""
    if not isinstance(val, (int, float)):
        return str(val)
    if np.isinf(val):
        return "inf"
    if np.isnan(val):
        return "nan"
    if val == 0:
        return "0.00"
    if abs(val) < 1e-5:
        return "< 1e-5"
    if abs(val) < 10:
        decimals = max(0, 1 - math.floor(math.log10(abs(val))))
        return f"{val:.{decimals}f}"
    return f"{val:.1f}"


def draw_annotated_tree(tree, values: dict, metric: str, title: str, paths) -> None:
    """
    Draw the tree as given (topology and branch lengths of the Newick file)
    with each tip labelled "<name> (<metric>: <value>)", N/A for tips without
    results. Writes PREFIX.<metric>.tree.png/svg.

    Args:
        tree: Bio.Phylo tree (not modified).
        values: Strain name -> metric value.
        metric: Column name, used in the labels and (via safe_name) file names.
        title: Figure title suffix.
        paths: Output paths from ``ensure_output_dirs``.
    """
    def label(clade):
        if not clade.is_terminal():
            return None
        value = format_value(values[clade.name]) if clade.name in values else "N/A"
        return f"{clade.name} ({metric}: {value})"

    n_tips = tree.count_terminals()
    fig, ax = plt.subplots(figsize=(20, max(8, 0.25 * n_tips)))
    Phylo.draw(tree, axes=ax, do_show=False, label_func=label)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_title(f"Phylogenetic Tree Annotated with {title}", fontsize=16)
    fig.tight_layout()

    stem = f"{safe_name(metric)}.tree"
    save_figure(fig, paths, stem, bbox_inches="tight", png_kwargs={"dpi": 300})
    log.info(f"Saved {metric} tree to {paths.png(stem)}")


def validate_tree(nwk: str, vcf: str):
    """
    Read the Newick tree and check that it shares samples with the VCF (only
    the VCF header is read). Exits on error or duplicate tip names; warns
    about partial overlap and a tree without branch lengths.

    Returns:
        The Bio.Phylo tree.
    """
    log.info(f"Validating tree file: {nwk}...")
    if not Path(nwk).exists():
        die(f"Tree file '{nwk}' does not exist.")

    try:
        tree = Phylo.read(nwk, "newick")
    except Exception as e:
        die(f"Invalid tree file: {e}")
    tip_names = [term.name for term in tree.get_terminals()]
    tree_samples = set(tip_names)

    if not tree_samples:
        die("Tree contains no terminal nodes (samples).")
    if len(tree_samples) < len(tip_names):
        dup = sorted({n for n in tip_names if tip_names.count(n) > 1}, key=str)
        die(f"Tree has duplicate tip names: {first_names(dup)}.")
    if not any(clade.branch_length for clade in tree.find_clades()):
        log.warning("WARNING: the tree has no branch lengths; it is drawn as a "
                    "cladogram.")

    log.info("Checking sample consistency...")
    vcf_samples, _, _ = read_vcf(vcf, fields=["samples"])
    vcf_samples_set = set(vcf_samples)

    common = tree_samples.intersection(vcf_samples_set)
    if not common:
        log.info(f"Tree samples (first 5): {list(tree_samples)[:5]}")
        log.info(f"VCF samples (first 5): {list(vcf_samples)[:5]}")
        die("No common samples found between Tree and VCF.")

    # Partial overlap is allowed: tips without results are labelled N/A,
    # VCF samples absent from the tree are only in the TSV output.
    missing_in_tree = vcf_samples_set - tree_samples
    missing_in_vcf = tree_samples - vcf_samples_set

    if missing_in_tree:
        log.warning(
            f"WARNING: {len(missing_in_tree)} samples in VCF are missing from "
            "the Tree."
        )
    if missing_in_vcf:
        log.warning(
            f"WARNING: {len(missing_in_vcf)} samples in Tree are missing from "
            "the VCF."
        )

    log.info("Tree validation successful.")
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate dN/dS and annotate tree.")
    parser.add_argument("--vcf", required=True, help="Input VCF")
    parser.add_argument("--gff", required=True, help="Input GFF")
    parser.add_argument("--fasta", required=True, help="Reference FASTA")
    parser.add_argument("--nwk", required=True, help="Newick tree")
    add_out_argument(parser)
    parser.add_argument(
        "--jobs", type=positive_int, default=default_jobs(),
        help="Worker processes (default: available CPUs minus one)",
    )
    args = parser.parse_args()
    setup_logging("selection", args)

    paths = ensure_output_dirs(args.out)

    tree = validate_tree(args.nwk, args.vcf)

    log.info("Loading reference and GFF...")
    ref_genome = load_reference_genome(args.fasta)
    genes = parse_gff(args.gff)

    genomes_data = reconstruct_genomes_from_vcf(args.vcf, ref_genome)
    n_genes = len(genes)
    gff_contigs = dict.fromkeys(c[0]["chrom"] for c in genes.values())
    genes = genes_in_vcf_region(genes, genomes_data[0])
    if not genes:
        die(f"none of the {n_genes} GFF genes lies in the region covered by the "
            f"VCF SNPs: check that the GFF sequence ids ({first_names(gff_contigs)}) "
            f"match the VCF contigs ({first_names(genomes_data[0])}).")
    if len(genes) < n_genes:
        log.info(
            f"Using {len(genes)} of {n_genes} genes inside the region covered by "
            "the VCF."
        )
    n_partial = count_partial_codons(genes)
    if n_partial:
        log.warning(f"Warning: {n_partial} genes have a CDS length (after the phase "
                    "trim) that is not a multiple of 3; the last partial codon is "
                    "ignored.")

    results_df, errors = run_genes(ref_genome, *genomes_data, genes, args.jobs)
    report_failures(errors, len(genes))

    # Save TSV (genes with at least one difference from the reference)
    out_tsv = paths.table("tsv")
    changed_df = results_df[(results_df["pN"] > 0) | (results_df["pS"] > 0)]
    write_table(changed_df, out_tsv, sep="\t", index=False)
    log.info(f"dN/dS results saved to {out_tsv}")
    if changed_df.empty:
        log.info("No gene differs from the reference in any sample.")

    strain_df = summarise_by_strain(results_df)
    write_table(strain_df, paths.table("summary.tsv"), sep="\t")

    values = strain_df.to_dict()
    for metric, title in (
        ("dS", "Genome-wide Synonymous Divergence (dS)"),
        ("dN", "Genome-wide Non-Synonymous Divergence (dN)"),
        ("dN/dS", "Genome-wide dN/dS Ratio"),
    ):
        draw_annotated_tree(tree, values[metric], metric, title, paths)


if __name__ == "__main__":
    main()
