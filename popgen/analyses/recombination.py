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
import hashlib
import math
import logging
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import NamedTuple

import numpy as np
import psutil

from popgen.cli import add_out_argument, fraction, positive_float, positive_int
from popgen.external import find_phi, phi_version
from popgen.genotypes import contig_rows, haploid_alleles
from popgen.io import atomic_write, ensure_output_dirs, read_vcf, safe_name
from popgen.runlog import die, setup_logging

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


def write_fasta(path, n_samples: int, h) -> None:
    """
    Write haplotypes as a SNP alignment: REF -> A, ALT -> C, missing -> N.

    Sequences are named s1, s2, ... in sample order: the names do not enter
    the tests, and real sample names may not fit Phi's C buffers.

    Args:
        path: FASTA file to write.
        n_samples: Number of samples (columns of h).
        h: Haploid alleles, sites x samples (0 REF, > 0 ALT, < 0 missing).
    """
    seqs = np.full(h.T.shape, "N", dtype="U1")
    seqs[h.T == 0] = "A"
    seqs[h.T > 0] = "C"  # biallelic: every ALT is one state
    with open(path, "w") as f:
        for i in range(n_samples):
            f.write(f">s{i + 1}\n{''.join(seqs[i])}\n")


def parse(text: str) -> dict:
    """Statistics and p-values from Phi's output (None where absent or "--").

    ``phi_mean`` / ``phi_var`` are the analytical values (first column of
    Phi's table), ``phi_mean_perm`` / ``phi_var_perm`` the permutation ones
    (second column)."""

    def grab(pat, group=1):
        m = re.search(pat, text, re.M)
        return m.group(group) if m else None

    two = r"^{}:\s+(\S+)\s+(\S+)"
    r = {
        "informative": grab(r"^Found (\d+) informative sites"),
        "phi_obs": grab(r"^Observed:\s+(\S+)"),
        "phi_mean": grab(r"^Mean:\s+(\S+)"),
        "phi_var": grab(r"^Variance:\s+(\S+)"),
        "phi_mean_perm": grab(two.format("Mean"), 2),
        "phi_var_perm": grab(two.format("Variance"), 2),
        "p_phi_normal": grab(r"^PHI \(Normal\):\s+(\S+)"),
        "p_phi_perm": grab(r"^PHI \(Permutation\):\s+(\S+)"),
        "p_nss": grab(r"^NSS:\s+(\S+)"),
        "p_maxchi": grab(r"^Max Chi\^2:\s+(\S+)"),
    }
    return {k: (None if v in (None, "--") else v) for k, v in r.items()}


def perm_count(text: str) -> int | None:
    """Number of permutations of Phi's PHI permutation test, or None."""
    m = re.search(r"^PHI \(Permutation\):\s+\S+\s+\((\d+) permutations\)", text, re.M)
    return int(m.group(1)) if m else None


def run_phi(phi, workdir, fasta, perms) -> tuple[int, str]:
    """Run Phi (with -o -v; -p perms if given); return (exit code, output)."""
    cmd = [phi, "-f", fasta, "-o", "-v"] + (["-p", str(perms)] if perms else [])
    p = subprocess.run(cmd, cwd=workdir, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


RERUN = "\n\n#### re-run with PHI permutation test ####\n"
ROW_STATS = ("phi_obs", "phi_mean", "phi_var", "p_phi_normal", "p_phi_perm",
             "p_nss", "p_maxchi")


class Job(NamedTuple):
    """One Phi test: a contig or a segment of one."""
    name: str  # contig, or contig:first_pos-last_pos for a segment
    safe_name: str  # unique; names the log file and work directory
    order: tuple[int, int]  # (contig index, first row): VCF order
    h: np.ndarray  # haploid alleles, sites x samples
    need: int  # estimated Phi memory (bytes)


class PhiRun(NamedTuple):
    """Exit code and output of one Phi run."""
    rc: int
    out: str


def unique_safe_names(names) -> list[str]:
    """``safe_name`` of each name, with _2, _3, ... appended to repeats in
    the given order (deterministic, so --resume finds the same logs)."""
    used, out = set(), []
    for n in names:
        base = s = safe_name(n)
        k = 1
        while s in used:
            k += 1
            s = f"{base}_{k}"
        used.add(s)
        out.append(s)
    return out


def plan_work(h, chrom, pos, step, min_snps, max_informative, max_ram):
    """
    Split the alignment into Phi tests: one per contig, or consecutive
    segments of a contig with more than ``max_informative`` informative sites.

    Args:
        h: Haploid alleles, sites x samples.
        chrom, pos: Contig and position of every site.
        step: Keep every step-th SNP per contig (``--subset``).
        min_snps: Skip contigs with fewer SNPs (after ``step``).
        max_informative: Most informative sites per test.
        max_ram: Bytes; tests that alone need more are not run.

    Returns:
        tuple: (jobs to run, jobs too large for max_ram, number of split
        contigs); jobs in VCF order.
    """
    planned, n_split = [], 0
    for ci, (c, rows) in enumerate(contig_rows(chrom, pos, "error").items()):
        rows = rows[::step]
        if len(rows) < min_snps:
            continue
        segs = split_segments(h[rows], max_informative)
        n_split += len(segs) > 1
        for a, b in segs:
            name = c if len(segs) == 1 else f"{c}:{pos[rows[a]]}-{pos[rows[b - 1]]}"
            hs = h[rows[a:b]]
            planned.append((name, (ci, a), hs, phi_bytes(informative(hs))))
    safe = unique_safe_names(p[0] for p in planned)
    jobs = [Job(name, s, order, hs, need)
            for (name, order, hs, need), s in zip(planned, safe)]
    return ([j for j in jobs if j.need <= max_ram],
            [j for j in jobs if j.need > max_ram], n_split)


def needs_rerun(first: PhiRun) -> bool:
    """True when Phi ran but its normal approximation is unavailable (too few
    informative sites for it), so PHI must be re-run as a permutation test."""
    r = parse(first.out)
    return (first.rc == 0 and r["p_phi_normal"] is None
            and r["informative"] is not None and int(r["informative"]) >= 2)


def interpret(first: PhiRun, rerun: PhiRun | None, perms: int | None) -> dict:
    """
    Result row of one test from Phi's output(s).

    Args:
        first: The first Phi run.
        rerun: The PHI permutation re-run, or None.
        perms: Permutations requested for the re-run (used when Phi's output
            does not state its count).

    Returns:
        dict: ``parse`` keys plus ``note`` and ``failed`` (a Phi run exited
        with an error, apart from the "fewer than 2 informative sites" case).
    """
    r = parse(first.out)
    note = "" if first.rc == 0 else f"Phi exit {first.rc}"
    failed = first.rc != 0
    if r["informative"] is not None and int(r["informative"]) < 2:
        # No site pairs to test: Phi prints NSS p = 0 for a NaN score, or crashes.
        r = {k: (v if k == "informative" else None) for k, v in r.items()}
        note, failed = "fewer than 2 informative sites", False
    elif rerun is not None:
        # NSS / Max Chi^2 stay from the first run.
        r2 = parse(rerun.out) if rerun.rc == 0 else {}
        p = r2.get("p_phi_perm")
        n = perm_count(rerun.out) or perms
        if p is not None and float(p) == 0:
            p = str(1 / n)  # 0 of N permutations: a bound, not zero
        r["p_phi_perm"] = p
        if p is not None:
            r["phi_mean"], r["phi_var"] = r2["phi_mean_perm"], r2["phi_var_perm"]
            note = f"PHI by {n} permutations"
        elif rerun.rc != 0:
            note, failed = f"Phi exit {rerun.rc} (PHI permutation re-run)", True
        else:
            note = "too few informative sites for PHI"
    r.update(note=note, failed=failed)
    return r


def alignment_header(h, n_samples: int) -> str:
    """First line of a Phi log: identifies the alignment for --resume."""
    a = np.ascontiguousarray(h, dtype=np.int8)
    digest = hashlib.sha256(a.tobytes())
    digest.update(repr(a.shape).encode())
    return (f"#popgen-phi alignment={digest.hexdigest()} samples={n_samples} "
            f"sites={a.shape[0]}")


def run_contig_test(job, n_samples, phi, tmp, logdir, perms_fallback, resume) -> dict:
    """
    Run Phi on one contig or segment and interpret its output; re-run PHI as
    a permutation test where the normal approximation is unavailable.

    With ``resume``, the output of an earlier run is reused when its log
    starts with the same alignment header (``alignment_header``); a re-run
    section is reused only if it used at least ``perms_fallback``
    permutations.

    Returns:
        dict: ``interpret``'s row plus ``contig``, ``snps`` and ``phi_runs``
        (Phi processes started).
    """
    d = Path(tmp) / job.safe_name
    log_path = Path(logdir) / f"{job.safe_name}.log"
    header = alignment_header(job.h, n_samples)
    prev = []
    if resume and log_path.exists():
        first_line, _, rest = log_path.read_text().partition("\n")
        if first_line == header:
            prev = rest.split(RERUN)
        elif first_line.startswith("#popgen-phi "):
            log.info(f"{job.name}: alignment changed since {log_path.name}; "
                     "re-running Phi.")
        else:
            log.info(f"{job.name}: {log_path.name} was written by an older PopGen "
                     "(no alignment header); not reused, re-running Phi.")
    started = 0

    def phi_run(i, perms):
        nonlocal started
        if len(prev) > i and "p-Value(s)" in prev[i] and (
                i == 0 or (perm_count(prev[i]) or 0) >= perms):
            return PhiRun(0, prev[i])  # same alignment (--resume)
        if not d.exists():
            d.mkdir()
            write_fasta(d / "in.fa", n_samples, job.h)
        started += 1
        return PhiRun(*run_phi(phi, d, "in.fa", perms))  # Phi writes Phi.* into its cwd

    first = phi_run(0, 0)
    rerun = phi_run(1, perms_fallback) if needs_rerun(first) else None
    with atomic_write(log_path) as f:
        f.write(header + "\n" + first.out + (RERUN + rerun.out if rerun else ""))
    shutil.rmtree(d, ignore_errors=True)
    r = interpret(first, rerun, perms_fallback)
    r.update(contig=job.name, snps=job.h.shape[0], phi_runs=started)
    return r


def holm(ps):
    """Holm step-down adjusted p-values; None stays None. Input must be finite."""
    assert all(p is None or math.isfinite(p) for p in ps), "non-finite p-value"
    idx = sorted((p, i) for i, p in enumerate(ps) if p is not None)
    adj, run, m = [None] * len(ps), 0.0, len(idx)
    for rank, (p, i) in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * p))
        adj[i] = run
    return adj


def num(x) -> float | None:
    """Phi's printed number as a float; None if absent or not finite."""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def finalise(rows) -> list[dict]:
    """
    Add ``p_phi`` (normal approximation, else permutation), ``phi_z`` and
    ``p_phi_holm`` (Holm over all rows with a p-value) to the rows, in place.
    A non-finite p-value is treated as missing and noted.
    """
    for r in rows:
        raw = r["p_phi_normal"] if r["p_phi_normal"] is not None else r["p_phi_perm"]
        r["p_phi"] = num(raw)
        if raw is not None and r["p_phi"] is None:
            msg = "Phi printed a non-finite p-value"
            r["note"] = f"{r['note']}; {msg}" if r.get("note") else msg
        o, m, v = num(r["phi_obs"]), num(r["phi_mean"]), num(r["phi_var"])
        # Standardised PHI: how far below the clonal expectation (negative =
        # incompatible sites closer together than expected = recombination).
        r["phi_z"] = (o - m) / math.sqrt(v) if None not in (o, m, v) and v > 0 else None
    for r, a in zip(rows, holm([r["p_phi"] for r in rows])):
        r["p_phi_holm"] = a
    return rows


COLUMNS = ["contig", "snps", "informative", "phi_obs", "phi_mean", "phi_z",
           "p_phi", "p_phi_holm", "p_nss", "p_maxchi", "note"]


def write_table(path, rows) -> None:
    """Write the per-test table (``COLUMNS``; NA for missing values)."""
    with atomic_write(path) as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            vals = []
            for c in COLUMNS:
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


def summary_lines(rows, *, vcf, n_samples, n_contigs, max_informative, n_split,
                  n_run, n_too_big, step, tsv_name, logdir_name) -> list[str]:
    """Lines of the summary file from the finalised rows and the run plan."""
    tested = [r for r in rows if r["p_phi"] is not None]
    failed = [r for r in rows if r.get("failed")]
    sig = [r for r in tested if r["p_phi_holm"] < 0.05]
    lo = min((r["p_phi_holm"] for r in tested), default=None)
    if tested:
        verdict = (f"Clonality (no recombination on any contig): "
                   f"{'rejected' if sig else 'not rejected'} "
                   f"(smallest Holm p = {lo:.3g})")
    else:
        verdict = "Clonality: not tested (no PHI p-value obtained)"
    lines = [
        f"VCF: {vcf}",
        f"Samples: {n_samples}; contigs: {n_contigs}; "
        f"split into segments of <= {max_informative} informative sites: "
        f"{n_split}; tests (contigs or segments) run: {n_run}; "
        f"too large for --max-ram: {n_too_big}; "
        f"with a PHI p-value: {len(tested)}",
        f"Phi runs failed: {len(failed)}",
        f"SNPs used: {'all' if step == 1 else f'every {step}th'} per contig"
        f"{'' if step == 1 else f' (fraction {1 / step:.3g})'}; "
        f"PHI window: 100 SNPs (Phi default)",
        f"Contigs or segments with significant PHI (Holm-adjusted p < 0.05): "
        f"{len(sig)} of {len(tested)}",
        verdict,
        f"Per contig: {tsv_name}; Phi output: {logdir_name}/",
    ]
    if lo == 0:
        lines.insert(-1, "p = 0: smaller than the smallest p that Phi's normal "
                         "approximation prints.")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="PhiPack recombination tests, per contig.")
    ap.add_argument("--vcf", required=True, help="Input VCF (haploid, biallelic)")
    add_out_argument(ap)
    ap.add_argument("--subset", type=fraction, default=1.0,
                    help="Keep every n-th SNP per contig, n = ceil(1/subset) "
                         "(default: all; only for a contig too large for RAM); "
                         "unlike ld_decay.py --subset, which draws SNPs at random")
    ap.add_argument("--jobs", type=positive_int, default=8,
                    help="Contigs run in parallel (at most)")
    ap.add_argument("--max-ram", type=positive_float, default=None,
                    help="GB that running Phi processes may use together "
                         "(default: 80%% of the memory available at start)")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse complete Phi logs of the same alignments in "
                         "<out>.phi.logs/")
    ap.add_argument("--min-snps", type=positive_int, default=2,
                    help="Skip contigs with fewer SNPs "
                         "(default 2: PHI needs site pairs)")
    ap.add_argument("--max-informative", type=positive_int, default=PHI_MAX_INFORMATIVE,
                    help="Split larger contigs into consecutive segments with at "
                         f"most this many informative sites (default "
                         f"{PHI_MAX_INFORMATIVE}: Phi's statistics overflow above)")
    args = ap.parse_args()
    if args.max_informative > PHI_MAX_INFORMATIVE:
        ap.error(f"--max-informative must be <= {PHI_MAX_INFORMATIVE} "
                 "(Phi's statistics overflow above)")
    setup_logging("recombination", args)

    phi = find_phi()
    if phi is None:
        die("PhiPack's 'Phi' program not found (install it with "
            "tools/install_phipack.py, see README.md).")
    log.info(f"Using {phi} (PhiPack version: {phi_version(phi) or 'unknown'})")

    paths = ensure_output_dirs(args.out)
    samples, gt, cs = read_vcf(args.vcf, fields=["samples", "calldata/GT",
                                                 "variants/CHROM", "variants/POS"])
    h = haploid_alleles(gt, "error")
    chrom, pos = cs["variants/CHROM"], cs["variants/POS"]
    if (h < 0).any():
        log.warning(f"Warning: {int((h < 0).sum())} missing calls written as N.")

    n_contigs = len(set(chrom))
    step = max(1, math.ceil(1 / args.subset - 1e-9))
    if step > 1:
        log.info(f"--subset {args.subset}: one SNP in {step} per contig is used "
                 f"(fraction {1 / step:.3g}).")
    max_ram = args.max_ram * GB if args.max_ram else 0.8 * mem_available()
    work, too_big, n_split = plan_work(h, chrom, pos, step, args.min_snps,
                                       args.max_informative, max_ram)
    # A permutation p is >= 1/perms; Holm needs m * p < 0.05, so perms > 20 m.
    perms = max(PERM_DEFAULT, 40 * len(work))

    logdir = paths.table("phi.logs")
    logdir.mkdir(exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="phi.", dir=paths.results)
    log.info(f"Testing {len(work)} contigs or segments of {n_contigs} contigs "
             f"({n_split} split; < {args.min_snps} SNPs skipped), "
             f"up to {args.jobs} at a time "
             f"within {max_ram / GB:.1f} GB (largest needs "
             f"{max((j.need for j in work), default=0) / GB:.1f} GB)...")
    try:
        res = run_scheduled(
            [(j.name, j, j.need) for j in work], args.jobs, max_ram,
            lambda w: run_contig_test(w[1], len(samples), phi, tmp, logdir, perms,
                                      args.resume))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    started = sum(r["phi_runs"] for r in res)
    log.info(f"Phi processes started: {started}"
             + (f" (other results reused from {logdir.name}/)" if args.resume else ""))
    for j in too_big:
        res.append(dict(contig=j.name, snps=j.h.shape[0],
                        informative=str(informative(j.h)),
                        **{k: None for k in ROW_STATS},
                        note=f"not run: Phi needs ~{j.need / GB:.1f} GB > --max-ram"))
    order = {j.name: j.order for j in work + too_big}
    res.sort(key=lambda r: order[r["contig"]])  # VCF order
    finalise(res)
    for r in res:
        if r.get("failed"):
            log.error(f"Phi failed on {r['contig']}: {r['note']}")

    tsv = paths.table("phi.contigs.tsv")
    write_table(tsv, res)
    lines = summary_lines(
        res, vcf=args.vcf, n_samples=len(samples), n_contigs=n_contigs,
        max_informative=args.max_informative, n_split=n_split, n_run=len(work),
        n_too_big=len(too_big), step=step, tsv_name=tsv.name, logdir_name=logdir.name)
    with atomic_write(paths.table("phi.summary.txt")) as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("\n".join(lines))
    n_failed = sum(bool(r.get("failed")) for r in res)
    if n_failed:
        die(f"{n_failed} Phi run(s) failed (see above and {logdir.name}/); "
            "the tables and summary were written for the other tests.", code=2)
    log.info("Recombination analysis complete.")


if __name__ == "__main__":
    main()
