"""
Recombination tests (PhiPack: PHI, NSS, Max Chi^2) run per contig.

PHI asks whether incompatible site pairs are closer together than expected
under clonality, so the order and adjacency of sites in the alignment must be
real. Each contig is therefore tested on its own SNP alignment, in genome
order; contigs are never concatenated (that created false adjacencies at
every junction and needed --subset to fit Phi's all-pairs matrix in RAM).
Windows (-w, default 100) are counted in alignment columns, i.e. SNPs here.

PHI is the primary test (Bruen et al. 2006): one p-value per contig,
Holm-adjusted across the contigs tested (valid under any dependence between
contigs), so "any contig significant after Holm" rejects clonality for the
genome. Where Phi's normal approximation is unavailable (too few informative
sites) PHI is re-run as a permutation test with enough permutations to reach
the Holm threshold. NSS and Max Chi^2 are reported as secondary evidence;
their 1000 permutations give p >= 0.001.

Max Chi^2's "best breakpoint" is not reported: PhiPack reports the centre of
the best-scoring window, which spans two thirds of the polymorphic sites, so
it does not locate a recombination event.

Phi holds an int matrix over all pairs of informative sites: RAM per contig
is about 4 * n^2 bytes (n informative sites; measured: 9366 -> 0.37 GB,
104172 -> 42.5 GB). Contigs are started largest first while their estimated
sum fits --max-ram; a contig that alone exceeds it is not run.

Phi's analytical PHI mean and variance are wrong above about 46,340
informative sites (46,341^2 > 2^31: an int overflow; observed: impossible
negative means, p = 0 or 1). A contig with more informative sites is split
into consecutive segments of at most --max-informative informative sites,
each tested on its own and named contig:first_pos-last_pos. Splitting keeps
the real order and adjacency of the SNPs; only pairs across a segment border
are not compared. Holm's adjustment runs over all tests (contigs and segments).

Input must be haploid (or homozygous diploid) genotypes; heterozygous calls
are refused, because unphased heterozygotes are not haplotypes.
"""

import argparse
import math
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import ensure_output_dirs, find_phi, read_vcf, setup_logging

log = logging.getLogger("recombination")

PERM_DEFAULT = 1000  # Phi's own NSS / Max Chi^2 permutation count
PHI_MAX_INFORMATIVE = 46340  # 46341^2 overflows Phi's 32-bit int
GB = 1024 ** 3


def informative_mask(h):
    """Phi's informative sites: both alleles in >= 2 sequences (missing ignored)."""
    return ((h == 0).sum(1) >= 2) & ((h > 0).sum(1) >= 2)


def informative(h) -> int:
    """Number of informative sites (see ``informative_mask``)."""
    return int(informative_mask(h).sum())


def split_segments(h, limit):
    """Split rows of h into the fewest consecutive (start, stop) row ranges with
    at most limit informative sites each, balanced in informative sites."""
    idx = np.flatnonzero(informative_mask(h))
    k = max(1, math.ceil(len(idx) / limit))
    cuts = [int(idx[(len(idx) * s) // k]) for s in range(1, k)]
    bounds = [0] + cuts + [h.shape[0]]
    return list(zip(bounds[:-1], bounds[1:]))


def phi_bytes(n: int) -> int:
    """Estimated Phi memory for n informative sites."""
    return 4 * n * n + 64 * 1024 ** 2  # pair matrix + overhead


def mem_available() -> int:
    """Bytes of memory currently available (psutil, portable)."""
    return psutil.virtual_memory().available


def run_scheduled(work, jobs, max_ram, fn):
    """Run fn(w) for each w = (contig, h, bytes), largest first, while the
    estimated bytes of running contigs fit max_ram. Returns results in work order."""
    res, cond, state = {}, threading.Condition(), {"jobs": 0, "ram": 0}

    def go(i, w):
        try:
            res[i] = fn(w)
        except BaseException as e:  # re-raised in the main thread
            res[i] = e
        finally:
            with cond:
                state["jobs"] -= 1
                state["ram"] -= w[2]
                cond.notify_all()

    threads = []
    for i in sorted(range(len(work)), key=lambda i: -work[i][2]):
        with cond:
            cond.wait_for(lambda: state["jobs"] < jobs
                          and state["ram"] + work[i][2] <= max_ram)
            state["jobs"] += 1
            state["ram"] += work[i][2]
        t = threading.Thread(target=go, args=(i, work[i]))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    for r in res.values():
        if isinstance(r, BaseException):
            raise r
    return [res[i] for i in range(len(work))]


def haplotypes(gt):
    """(n_vars, n_samples) allele matrix; -1 = missing. Refuses heterozygotes."""
    g = np.asarray(gt)
    if g.ndim == 3 and g.shape[2] > 1:
        a, b = g[:, :, 0], g[:, :, 1]
        het = (a >= 0) & (b >= 0) & (a != b)
        if het.any():
            sys.exit(
                f"Error: {int(het.sum())} heterozygous genotype calls. PhiPack "
                "needs haplotypes; phase the data or use haploid calls."
            )
        return a
    return g.reshape(g.shape[0], g.shape[1])


def write_fasta(path, samples, h) -> None:
    """Write haplotypes as a SNP alignment: REF -> A, ALT -> C, missing -> N."""
    seqs = np.full(h.T.shape, "N", dtype="U1")
    seqs[h.T == 0] = "A"
    seqs[h.T > 0] = "C"  # biallelic: every ALT is one state
    with open(path, "w") as f:
        for i, s in enumerate(samples):
            f.write(f">{s}\n{''.join(seqs[i])}\n")


def parse(text: str) -> dict:
    """Statistics and p-values from Phi's output (None where absent or "--")."""

    def grab(pat):
        m = re.search(pat, text, re.M)
        return m.group(1) if m else None

    r = {
        "informative": grab(r"^Found (\d+) informative sites"),
        "phi_obs": grab(r"^Observed:\s+(\S+)"),
        "phi_mean": grab(r"^Mean:\s+(\S+)"),
        "phi_var": grab(r"^Variance:\s+(\S+)"),
        "p_phi_normal": grab(r"^PHI \(Normal\):\s+(\S+)"),
        "p_phi_perm": grab(r"^PHI \(Permutation\):\s+(\S+)"),
        "p_nss": grab(r"^NSS:\s+(\S+)"),
        "p_maxchi": grab(r"^Max Chi\^2:\s+(\S+)"),
    }
    return {k: (None if v in (None, "--") else v) for k, v in r.items()}


PHI = "Phi"  # set to the resolved path in main()


def run_phi(workdir, fasta, perms) -> tuple[int, str]:
    """Run Phi (with -o -v; -p perms if given); return (exit code, output)."""
    cmd = [PHI, "-f", fasta, "-o", "-v"] + (["-p", str(perms)] if perms else [])
    p = subprocess.run(cmd, cwd=workdir, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


RERUN = "\n\n#### re-run with PHI permutation test ####\n"


def test_contig(contig, samples, h, tmp, logdir, perms_fallback, resume) -> dict:
    """
    Run Phi on one contig or segment and parse its output; re-run PHI as a
    permutation test where the normal approximation is unavailable. With
    ``resume``, a complete earlier log of the same alignment is reused.
    """
    name = re.sub(r"[^\w.-]", "_", contig)
    d = os.path.join(tmp, name)
    log = os.path.join(logdir, f"{name}.log")
    prev = open(log).read().split(RERUN) if resume and os.path.exists(log) else []
    same = f"Found {len(samples)} sequences of length {h.shape[0]}"

    def phi(i, perms):
        if len(prev) > i and same in prev[i] and "p-Value(s)" in prev[i]:
            return 0, prev[i]  # complete output of this same alignment (--resume)
        if not os.path.exists(os.path.join(d, "in.fa")):
            os.makedirs(d)
            write_fasta(os.path.join(d, "in.fa"), samples, h)
        return run_phi(d, "in.fa", perms)  # Phi writes Phi.* into its cwd

    rc, out = phi(0, 0)
    r = parse(out)
    note = "" if rc == 0 else f"Phi exit {rc}"
    if r["informative"] is not None and int(r["informative"]) < 2:
        # No site pairs to test: Phi prints NSS p = 0 for a NaN score, or crashes.
        r = {k: (v if k == "informative" else None) for k, v in r.items()}
        note = "fewer than 2 informative sites"
    elif rc == 0 and r["p_phi_normal"] is None and r["informative"] not in (None, "0"):
        # Too few informative sites for the normal approximation (Phi's advice:
        # use its permutation test). NSS / Max Chi^2 stay from the first run.
        rc2, out2 = phi(1, perms_fallback)
        out += RERUN + out2
        p = parse(out2)["p_phi_perm"] if rc2 == 0 else None
        if p is not None and float(p) == 0:
            p = str(1 / perms_fallback)  # 0 of N permutations: a bound, not zero
        r["p_phi_perm"] = p
        note = (f"PHI by {perms_fallback} permutations" if p is not None
                else "too few informative sites for PHI")
    with open(log, "w") as f:
        f.write(out)
    shutil.rmtree(d, ignore_errors=True)
    r.update(contig=contig, snps=h.shape[0], note=note)
    return r


def holm(ps):
    """Holm step-down adjusted p-values; None stays None."""
    idx = sorted((p, i) for i, p in enumerate(ps) if p is not None)
    adj, run, m = [None] * len(ps), 0.0, len(idx)
    for rank, (p, i) in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * p))
        adj[i] = run
    return adj


def main() -> None:
    global PHI
    ap = argparse.ArgumentParser(description="PhiPack recombination tests, per contig.")
    ap.add_argument("--vcf", required=True, help="Input VCF (haploid, biallelic)")
    ap.add_argument("--out", required=True, help="Output prefix")
    ap.add_argument("--subset", type=float, default=1.0,
                    help="Keep every 1/subset-th SNP per contig (default: all; "
                         "only for a contig too large for RAM)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="Contigs run in parallel (at most)")
    ap.add_argument("--max-ram", type=float, default=None,
                    help="GB that running Phi processes may use together "
                         "(default: 80%% of the memory available at start)")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse complete Phi logs of the same alignments in "
                         "<out>.phi.logs/")
    ap.add_argument("--min-snps", type=int, default=2,
                    help="Skip contigs with fewer SNPs "
                         "(default 2: PHI needs site pairs)")
    ap.add_argument("--max-informative", type=int, default=PHI_MAX_INFORMATIVE,
                    help="Split larger contigs into consecutive segments with at "
                         f"most this many informative sites (default "
                         f"{PHI_MAX_INFORMATIVE}: Phi's statistics overflow above)")
    args = ap.parse_args()
    setup_logging("recombination", args)

    PHI = find_phi()
    if PHI is None:
        sys.exit("Error: PhiPack's 'Phi' program not found (install it with "
                 "tools/install_phipack.py, see README.md).")
    log.info(f"Using {PHI}")

    paths = ensure_output_dirs(args.out)
    samples, gt, cs = read_vcf(args.vcf, fields=["samples", "calldata/GT",
                                                 "variants/CHROM", "variants/POS"])
    h = haplotypes(gt)
    chrom, pos = cs["variants/CHROM"], cs["variants/POS"]
    if (h < 0).any():
        log.warning(f"Warning: {int((h < 0).sum())} missing calls written as N.")

    contigs = list(dict.fromkeys(chrom))  # VCF order
    step = int(round(1 / args.subset)) if args.subset < 1 else 1
    max_ram = args.max_ram * GB if args.max_ram else 0.8 * mem_available()
    work, too_big, order, n_split = [], [], {}, 0
    for ci, c in enumerate(contigs):
        rows = np.flatnonzero(chrom == c)[::step]
        if len(rows) < args.min_snps:
            continue
        segs = split_segments(h[rows], args.max_informative)
        n_split += len(segs) > 1
        for a, b in segs:
            name = c if len(segs) == 1 else f"{c}:{pos[rows[a]]}-{pos[rows[b - 1]]}"
            order[name] = (ci, a)
            hs = h[rows[a:b]]
            need = phi_bytes(informative(hs))
            (work if need <= max_ram else too_big).append((name, hs, need))
    # A permutation p is >= 1/perms; Holm needs m * p < 0.05, so perms > 20 m.
    perms = max(PERM_DEFAULT, 40 * len(work))

    logdir = f"{paths['base']}.phi.logs"
    os.makedirs(logdir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="phi.", dir=os.path.dirname(paths["base"]))
    log.info(f"Testing {len(work)} contigs or segments of {len(contigs)} contigs "
             f"({n_split} split; < {args.min_snps} SNPs skipped), "
             f"up to {args.jobs} at a time "
             f"within {max_ram / GB:.1f} GB (largest needs "
             f"{max((w[2] for w in work), default=0) / GB:.1f} GB)...")
    try:
        res = run_scheduled(
            work, args.jobs, max_ram,
            lambda w: test_contig(w[0], samples, w[1], tmp, logdir, perms,
                                  args.resume))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for c, hc, need in too_big:
        res.append(dict(contig=c, snps=hc.shape[0], informative=str(informative(hc)),
                        **{k: None for k in ("phi_obs", "phi_mean", "phi_var",
                                             "p_phi_normal", "p_phi_perm", "p_nss",
                                             "p_maxchi")},
                        note=f"not run: Phi needs ~{need / GB:.1f} GB > --max-ram"))
    res.sort(key=lambda r: order[r["contig"]])  # VCF order

    def num(x):
        return float(x) if x is not None else None

    for r in res:
        r["p_phi"] = num(r["p_phi_normal"]) if r["p_phi_normal"] is not None \
            else num(r["p_phi_perm"])
        o, m, v = num(r["phi_obs"]), num(r["phi_mean"]), num(r["phi_var"])
        # Standardised PHI: how far below the clonal expectation (negative =
        # incompatible sites closer together than expected = recombination).
        r["phi_z"] = (o - m) / math.sqrt(v) if None not in (o, m, v) and v > 0 else None
    for r, a in zip(res, holm([r["p_phi"] for r in res])):
        r["p_phi_holm"] = a

    cols = ["contig", "snps", "informative", "phi_obs", "phi_mean", "phi_z",
            "p_phi", "p_phi_holm", "p_nss", "p_maxchi", "note"]
    tsv = f"{paths['tsv']}.phi.contigs.tsv"
    with open(tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in res:
            vals = []
            for c in cols:
                x = r.get(c)
                if c in ("p_nss", "p_maxchi") and x is not None and float(x) == 0:
                    x = f"<{1 / PERM_DEFAULT:g}"  # 0 of 1000 permutations
                elif (c in ("phi_obs", "phi_mean", "p_nss", "p_maxchi")
                      and x is not None):
                    x = f"{float(x) + 0.0:.3g}"  # + 0.0 turns Phi's -0 into 0
                elif isinstance(x, float):
                    x = f"{x:.3g}"
                vals.append("NA" if x is None else str(x))
            fh.write("\t".join(vals) + "\n")

    tested = [r for r in res if r["p_phi"] is not None]
    sig = [r for r in tested if r["p_phi_holm"] < 0.05]
    lo = min((r["p_phi_holm"] for r in tested), default=float("nan"))
    lines = [
        f"VCF: {args.vcf}",
        f"Samples: {len(samples)}; contigs: {len(contigs)}; "
        f"split into segments of <= {args.max_informative} informative sites: "
        f"{n_split}; tests (contigs or segments) run: {len(work)}; "
        f"too large for --max-ram: {len(too_big)}; "
        f"with a PHI p-value: {len(tested)}",
        f"SNPs used: {'all' if step == 1 else f'every {step}th'} per contig; "
        f"PHI window: 100 SNPs (Phi default)",
        f"Contigs or segments with significant PHI (Holm-adjusted p < 0.05): "
        f"{len(sig)} of {len(tested)}",
        f"Clonality (no recombination on any contig): "
        f"{'rejected' if sig else 'not rejected'} (smallest Holm p = {lo:.3g})",
        f"Per contig: {os.path.basename(tsv)}; Phi output: {os.path.basename(logdir)}/",
    ]
    if lo == 0:
        lines.insert(-1, "p = 0: smaller than the smallest p that Phi's normal "
                         "approximation prints.")
    with open(f"{paths['tsv']}.phi.summary.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("\n".join(lines))
    log.info("Recombination analysis complete.")


if __name__ == "__main__":
    main()
