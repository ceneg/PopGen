"""
dN/dS of every sample against the reference (Nei-Gojobori 1986, Jukes-Cantor
corrected), per gene and pooled genome-wide, with the genome-wide values drawn
on a tree.
"""

import argparse
import copy
import logging
import math
import multiprocessing as mp
import os
import sys
import warnings
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO
from Bio.Seq import Seq
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dnds import calculate_pairwise_dnds, jukes_cantor
from src.utils import ensure_output_dirs, read_vcf, setup_logging

log = logging.getLogger("selection")


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
        log.error(f"Error loading reference genome: {e}")
        sys.exit(1)
    if not ref_genome:
        log.error(f"Error: no sequences found in {fasta_path}")
        sys.exit(1)
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
        log.error(f"Error parsing GFF: {e}")
        sys.exit(1)


def reconstruct_genomes_from_vcf(vcf_path: str, ref_genome: dict[str, str]) -> tuple:
    """
    Collect the SNPs needed to reconstruct each sample's coding sequences.

    Only single-base REF/ALT variants are used (indels would shift the
    reading frame), and only those whose REF matches the reference FASTA.
    Uses the first allele of each call (haploid data, or homozygous diploid).

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

    gt_hap = np.asarray(gt[:, :, 0])  # Haploid or first allele

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
    for chrom in dict.fromkeys(chroms):  # VCF order
        idx = np.flatnonzero((chroms == chrom) & is_snp)
        if chrom not in ref_genome:
            n_off_contig += len(idx)
            continue
        idx = idx[np.argsort(poss[idx], kind="stable")]
        seq = ref_genome[chrom]
        in_range = poss[idx] < len(seq)
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
                    "dN/dS": dN / dS if dS > 0 else np.nan,
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


def calculate_dnds_for_genomes(ref_genome, genomes_data, genes) -> pd.DataFrame:
    """
    Count every gene in parallel on all CPU cores but one.

    Returns:
        DataFrame with one row per gene and sample (all genes).
    """
    variants_by_chrom, samples = genomes_data

    processes = max(1, mp.cpu_count() - 1)
    log.info(f"Calculating dN/dS using {processes} parallel processes...")

    items = list(genes.items())
    total_genes = len(items)
    chunksize = 20

    all_results = []
    errors = []

    # With fork, workers inherit the (read-only) data copy-on-write instead of
    # each receiving a pickled copy; other start methods pass it explicitly.
    init_worker(ref_genome, variants_by_chrom, samples)
    if "fork" in mp.get_all_start_methods():
        pool = mp.get_context("fork").Pool(processes=processes)
    else:
        pool = mp.Pool(
            processes=processes,
            initializer=init_worker,
            initargs=(ref_genome, variants_by_chrom, samples),
        )
    with pool:
        count = 0
        for res, err in pool.imap_unordered(process_gene, items, chunksize=chunksize):
            all_results.extend(res)
            if err:
                errors.append(err)
            count += 1
            if count % 500 == 0:
                log.info(f"Processed {count}/{total_genes} genes...")

    log.info(f"Finished processing {total_genes} genes.")
    if errors:
        log.warning(f"Warning: {len(errors)} genes failed and were skipped, "
                    f"e.g. {errors[0]}")
    return pd.DataFrame(all_results)


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
    with np.errstate(invalid="ignore", divide="ignore"):
        strain_df["dN/dS"] = strain_df["dN"] / strain_df["dS"]
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


def annotate_tree(tree, df: pd.DataFrame, metric_col: str):
    """
    Copy of ``tree`` with each tip renamed to "<name> (<metric>: <value>)"
    (N/A for tips without results).
    """
    tree_copy = copy.deepcopy(tree)
    label_map = df.set_index("strain")[metric_col].to_dict()
    for clade in tree_copy.get_terminals():
        name = clade.name
        value = format_value(label_map[name]) if name in label_map else "N/A"
        clade.name = f"{name} ({metric_col}: {value})"
    return tree_copy


def plot_and_save_dendrogram(tree, df, paths, metric_col, title_suffix) -> None:
    """
    Draw the annotated tree as a dendrogram, redrawn by UPGMA from its
    tip-to-tip distances (identical to the input for an ultrametric tree).
    """
    annotated_tree = annotate_tree(tree, df, metric_col)
    terminals = annotated_tree.get_terminals()
    labels = [term.name for term in terminals]
    n = len(labels)

    if n == 0:
        return

    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = annotated_tree.distance(terminals[i], terminals[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    linkage_matrix = linkage(squareform(dist_matrix), method="average")

    plt.figure(figsize=(20, max(8, 0.25 * n)))
    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=0,
        orientation="left",
        leaf_font_size=12,
    )
    plt.gca().spines[["top", "right", "left", "bottom"]].set_visible(False)
    plt.title(f"Phylogenetic Tree Annotated with {title_suffix}", fontsize=16)
    plt.tight_layout()

    safe_metric = metric_col.replace("/", "_")
    png_file = f"{paths['png']}.{safe_metric}.upgma.png"
    svg_file = f"{paths['svg']}.{safe_metric}.upgma.svg"

    plt.savefig(png_file, bbox_inches="tight", dpi=300)
    plt.savefig(svg_file, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {safe_metric} tree to {png_file}")


def validate_tree(nwk: str, vcf: str):
    """
    Read the Newick tree and check that it shares samples with the VCF (only
    the VCF header is read). Exits on error; warns about partial overlap.

    Returns:
        The Bio.Phylo tree.
    """
    log.info(f"Validating tree file: {nwk}...")
    if not os.path.exists(nwk):
        log.error(f"ERROR: Tree file '{nwk}' does not exist.")
        sys.exit(1)

    try:
        tree = Phylo.read(nwk, "newick")
        tree_samples = {term.name for term in tree.get_terminals()}

        if not tree_samples:
            log.error("ERROR: Tree contains no terminal nodes (samples).")
            sys.exit(1)

        log.info("Checking sample consistency...")
        vcf_samples, _, _ = read_vcf(vcf, fields=["samples"])
        vcf_samples_set = set(vcf_samples)

        common = tree_samples.intersection(vcf_samples_set)
        if not common:
            log.error("ERROR: No common samples found between Tree and VCF.")
            log.info(f"Tree samples (first 5): {list(tree_samples)[:5]}")
            log.info(f"VCF samples (first 5): {list(vcf_samples)[:5]}")
            sys.exit(1)

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

    except Exception as e:
        log.error(f"ERROR: Invalid tree file: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate dN/dS and annotate tree.")
    parser.add_argument("--vcf", required=True, help="Input VCF")
    parser.add_argument("--gff", required=True, help="Input GFF")
    parser.add_argument("--fasta", required=True, help="Reference FASTA")
    parser.add_argument("--nwk", required=True, help="Newick tree")
    parser.add_argument("--out", required=True, help="Output prefix")
    args = parser.parse_args()
    setup_logging("selection", args)

    paths = ensure_output_dirs(args.out)

    validate_tree(args.nwk, args.vcf)

    log.info("Loading reference and GFF...")
    ref_genome = load_reference_genome(args.fasta)
    genes = parse_gff(args.gff)

    genomes_data = reconstruct_genomes_from_vcf(args.vcf, ref_genome)
    n_genes = len(genes)
    genes = genes_in_vcf_region(genes, genomes_data[0])
    if len(genes) < n_genes:
        log.info(
            f"Using {len(genes)} of {n_genes} genes inside the region covered by "
            "the VCF."
        )

    results_df = calculate_dnds_for_genomes(ref_genome, genomes_data, genes)

    # Save TSV (genes with at least one difference from the reference)
    out_tsv = f"{paths['tsv']}.tsv"
    if results_df.empty:
        changed_df = results_df
    else:
        changed_df = results_df[(results_df["pN"] > 0) | (results_df["pS"] > 0)]
    changed_df.to_csv(out_tsv, sep="\t", index=False)
    log.info(f"dN/dS results saved to {out_tsv}")

    if not changed_df.empty:
        strain_df = summarise_by_strain(results_df)
        strain_df.to_csv(f"{paths['tsv']}.summary.tsv", sep="\t")

        try:
            tree = Phylo.read(args.nwk, "newick")
            for metric, title in (
                ("dS", "Genome-wide Synonymous Divergence (dS)"),
                ("dN", "Genome-wide Non-Synonymous Divergence (dN)"),
                ("dN/dS", "Genome-wide dN/dS Ratio"),
            ):
                plot_and_save_dendrogram(
                    tree, strain_df.reset_index(), paths, metric, title
                )

        except Exception as e:
            # Non-fatal: the TSV results are already written
            log.error(f"ERROR: Failed to process tree for visualization: {e}")

    else:
        log.info("Skipping visualization (no results).")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)  # Biopython warnings
    main()
