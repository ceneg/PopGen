import numpy as np
import pytest


@pytest.fixture(scope="module")
def rec(load_script):
    return load_script("recombination")


def test_holm(rec):
    assert rec.holm([0.01, None, 0.04, 0.03]) == pytest.approx([0.03, None, 0.06, 0.06])


def test_parse_phi_output(rec):
    out = "Found 12 informative sites.\nPHI (Normal):        1.2e-03\nNSS:  --\n"
    r = rec.parse(out)
    assert r["informative"] == "12"
    assert r["p_phi_normal"] == "1.2e-03"
    assert r["p_nss"] is None


def test_split_segments(rec):
    rng = np.random.default_rng(3)
    h = rng.integers(0, 2, (1000, 12))
    h[::7] = 0  # monomorphic rows, not informative
    n_inf = rec.informative(h)
    assert rec.split_segments(h, n_inf) == [(0, 1000)]
    segs = rec.split_segments(h, 100)
    assert len(segs) == -(-n_inf // 100)  # fewest segments
    assert segs[0][0] == 0 and segs[-1][1] == 1000
    assert all(a[1] == b[0] for a, b in zip(segs, segs[1:]))  # consecutive
    sizes = [rec.informative(h[a:b]) for a, b in segs]
    assert sum(sizes) == n_inf and max(sizes) <= 100 and max(sizes) - min(sizes) <= 1


DATA = __import__("conftest").PROJECT_ROOT / "tests" / "data" / "phi"


def test_holm_skips_none(rec):
    assert rec.holm([0.001, None, 0.01, 0.04]) == pytest.approx(
        [0.003, None, 0.02, 0.04])


def test_holm_refuses_nan(rec):
    """T-REC-1 (B6). v1.01: [0.004, 1.0, 1.0, 1.0], one NaN erased every later rank."""
    with pytest.raises((AssertionError, ValueError)):
        rec.holm([0.001, float("nan"), 0.01, 0.04])


def test_parse_canned_outputs(rec):
    r = rec.parse((DATA / "ctg1.log").read_text())
    assert r["informative"] == "254"
    assert r["p_phi_normal"] == "0.00e+00"
    assert r["phi_mean"] == "5.78e-01" and r["phi_var"] == "1.93e-05"
    first, rerun = (DATA / "ctg2.log").read_text().split(rec.RERUN)
    r = rec.parse(first)
    assert r["informative"] == "164" and r["p_phi_normal"] is None
    assert r["p_nss"] == "0.00e+00"
    assert rec.parse(rerun)["p_phi_perm"] == "0.00e+00"
    r = rec.parse((DATA / "truncated.log").read_text())
    assert r["informative"] == "254" and r["p_phi_normal"] is None


def test_run_scheduled_respects_ram_and_order(rec):
    import threading
    import time

    lock = threading.Lock()
    running, peak, events = [0], [0], []

    def fn(w):
        with lock:
            running[0] += w[2]
            peak[0] = max(peak[0], running[0])
            events.append(("start", w[0]))
        time.sleep(0.05)
        with lock:
            running[0] -= w[2]
            events.append(("stop", w[0]))
        return w[0]

    work = [("a", None, 6), ("b", None, 5), ("c", None, 4)]
    assert rec.run_scheduled(work, 2, 10, fn) == ["a", "b", "c"]
    assert peak[0] <= 10
    assert len(events) == 6


def test_run_scheduled_reraises_after_others_finish(rec):
    done = []

    def fn(w):
        if w[0] == "bad":
            raise RuntimeError("boom")
        done.append(w[0])
        return w[0]

    work = [("bad", None, 3), ("x", None, 2), ("y", None, 1)]
    with pytest.raises(RuntimeError, match="boom"):
        rec.run_scheduled(work, 2, 10, fn)
    assert sorted(done) == ["x", "y"]


@pytest.mark.parametrize("option", [
    ["--jobs", "0"],
    ["--max-informative", "0"],
    ["--max-informative", "50000"],
    ["--subset", "0"],
])
def test_bad_numeric_option_is_usage_error(tmp_path, option):
    """T-REC-6 (B15): rejected before any work (--jobs 0 used to hang)."""
    from conftest import PROJECT_ROOT, run_script

    vcf = PROJECT_ROOT / "examples" / "data" / "example.vcf"
    result = run_script("recombination", tmp_path, "--vcf", vcf,
                        "--out", tmp_path / "rec", *option, timeout=60)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not (tmp_path / "results").exists()


def row(rec, contig, p_normal, **kw):
    r = {k: None for k in rec.ROW_STATS}
    r.update(contig=contig, snps=10, informative="5", note="", p_phi_normal=p_normal)
    r.update(kw)
    return r


def test_finalise_nan_row_leaves_others(rec):
    """T-REC-1 (B6): a NaN p-value is missing, noted, and does not touch Holm."""
    ps = ["0.001", "nan", "0.01", "0.04"]
    rows = rec.finalise([row(rec, f"c{i}", p) for i, p in enumerate(ps)])
    without = rec.finalise([row(rec, f"c{i}", p) for i, p in enumerate(ps) if i != 1])
    assert rows[1]["p_phi"] is None and rows[1]["p_phi_holm"] is None
    assert "non-finite p-value" in rows[1]["note"]
    assert [r["p_phi_holm"] for r in rows if r["contig"] != "c1"] == pytest.approx(
        [r["p_phi_holm"] for r in without])


def test_interpret_canned_outputs(rec):
    """T-REC-2 (B6): normal approximation, permutation fallback, killed run."""
    r = rec.interpret(rec.PhiRun(0, (DATA / "ctg1.log").read_text()), None, None)
    assert r["p_phi_normal"] == "0.00e+00" and r["phi_mean"] == "5.78e-01"
    assert r["note"] == "" and not r["failed"]

    first, rerun = (DATA / "ctg2.log").read_text().split(rec.RERUN)
    assert rec.needs_rerun(rec.PhiRun(0, first))
    r = rec.interpret(rec.PhiRun(0, first), rec.PhiRun(0, rerun), 1000)
    assert float(r["p_phi_perm"]) == pytest.approx(1 / 1000)
    assert r["phi_mean"] == "5.27e-01" and r["phi_var"] == "2.18e-05"
    assert r["note"] == "PHI by 1000 permutations" and not r["failed"]
    # the bound uses the count Phi reports, not the one requested
    r = rec.interpret(rec.PhiRun(0, first), rec.PhiRun(0, rerun), 5000)
    assert float(r["p_phi_perm"]) == pytest.approx(1 / 1000)

    killed = rec.PhiRun(-9, (DATA / "truncated.log").read_text())
    assert not rec.needs_rerun(killed)
    r = rec.interpret(killed, None, None)
    assert all(r[k] is None for k in rec.ROW_STATS)
    assert r["note"] == "Phi exit -9" and r["failed"]
    r = rec.interpret(rec.PhiRun(0, first), rec.PhiRun(-11, ""), 1000)
    assert r["p_phi_perm"] is None and r["failed"]


def test_interpret_fewer_than_two_informative_is_not_a_failure(rec):
    r = rec.interpret(rec.PhiRun(1, "Found 1 informative sites.\nNSS: 0.00e+00\n"),
                      None, None)
    assert r["p_nss"] is None and r["note"] == "fewer than 2 informative sites"
    assert not r["failed"]


SUMMARY_KW = dict(vcf="x.vcf", n_samples=4, n_contigs=2, max_informative=100,
                  n_split=0, n_run=2, n_too_big=0, step=1,
                  tsv_name="x.tsv", logdir_name="x.logs")


def test_summary_lines_not_tested(rec):
    """T-REC-3 (B6): no p-value is "not tested", not a negative result."""
    rows = rec.finalise([row(rec, "a", None, note="Phi exit -9", failed=True),
                         row(rec, "b", None)])
    text = "\n".join(rec.summary_lines(rows, **SUMMARY_KW))
    assert "Clonality: not tested (no PHI p-value obtained)" in text
    assert "not rejected" not in text
    assert "Phi runs failed: 1" in text


def test_summary_lines_rejected(rec):
    rows = rec.finalise([row(rec, "a", "0.001"), row(rec, "b", "0.5")])
    text = "\n".join(rec.summary_lines(rows, **SUMMARY_KW))
    assert "Phi runs failed: 0" in text
    assert "Clonality (no recombination on any contig): rejected" in text
    assert "1 of 2" in text


FAKE_PHI = """#!{python}
import sys
from pathlib import Path
first, rerun = Path({log!r}).read_text().split({sep!r})
with open({counter!r}, "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
print(rerun if "-p" in sys.argv else first)
"""


@pytest.fixture
def fake_phi(rec, tmp_path):
    """A Phi stand-in printing the canned fallback-shape output (ctg2.log) and
    recording each call in a counter file."""
    import os
    import stat
    import sys

    counter = tmp_path / "calls.txt"
    phi = tmp_path / "Phi"
    phi.write_text(FAKE_PHI.format(python=sys.executable, log=str(DATA / "ctg2.log"),
                                   sep=rec.RERUN, counter=str(counter)))
    phi.chmod(phi.stat().st_mode | stat.S_IXUSR)
    assert os.access(phi, os.X_OK)

    def calls():
        return counter.read_text().splitlines() if counter.exists() else []

    return str(phi), calls


def make_job(rec, h, name="ctg2"):
    return rec.Job(name, rec.safe_name(name), (0, 0), h, 0)


def test_run_contig_test_resume(rec, tmp_path, fake_phi):
    """T-REC-4 (B7, B17): reuse only a log of the same alignment."""
    phi, calls = fake_phi
    logdir, tmp = tmp_path / "logs", tmp_path / "work"
    logdir.mkdir()
    tmp.mkdir()
    h = np.random.default_rng(1).integers(0, 2, (30, 6))

    def run(h, resume, perms=1000):
        return rec.run_contig_test(make_job(rec, h), 6, phi, tmp, logdir, perms, resume)

    r = run(h, False)
    assert len(calls()) == 2 and r["phi_runs"] == 2  # first run + permutation re-run
    assert r["note"] == "PHI by 1000 permutations"
    log_text = (logdir / "ctg2.log").read_text()
    assert log_text.startswith(rec.alignment_header(h, 6) + "\n")
    assert not list(tmp.iterdir())  # work directory removed

    r = run(h, True)
    assert len(calls()) == 2 and r["phi_runs"] == 0  # all reused
    assert r["note"] == "PHI by 1000 permutations"

    r = run(h, True, perms=2000)  # reused re-run had too few permutations
    assert len(calls()) == 3 and r["phi_runs"] == 1
    assert calls()[-1].endswith("-p 2000")

    h2 = h.copy()
    h2[0, 0] = 1 - h2[0, 0]  # one genotype changed, same shape
    assert rec.alignment_header(h2, 6) != rec.alignment_header(h, 6)
    r = run(h2, True)
    assert len(calls()) == 5 and r["phi_runs"] == 2

    # a v1.01 log (no header) of the same alignment is not reused
    (logdir / "ctg2.log").write_text((DATA / "ctg2.log").read_text())
    r = run(h2, True)
    assert len(calls()) == 7 and r["phi_runs"] == 2


def test_plan_work_unique_safe_names(rec):
    """T-REC-5 (B17): contigs whose safe names collide get _2, deterministically."""
    h = np.random.default_rng(2).integers(0, 2, (40, 6))
    chrom = np.array(["a|1"] * 20 + ["a_1"] * 20)
    pos = np.tile(np.arange(1, 21), 2)
    names = []
    for _ in range(2):
        work, too_big, n_split = rec.plan_work(h, chrom, pos, 1, 2, 1000, 1e12)
        names.append([(j.name, j.safe_name) for j in work])
    assert names[0] == names[1] == [("a|1", "a_1"), ("a_1", "a_1_2")]
    assert too_big == [] and n_split == 0


def test_plan_work_segment_names_and_ram(rec):
    """T-REC-8: a split contig is named c:<first_pos>-<last_pos> per segment."""
    rng = np.random.default_rng(4)
    h = rng.integers(0, 2, (60, 8))
    h[:, :4] = [0, 0, 1, 1]  # every site informative
    chrom = np.array(["c"] * 40 + ["d"] * 20)
    pos = np.concatenate([np.arange(101, 141), np.arange(1, 21)])
    work, too_big, n_split = rec.plan_work(h, chrom, pos, 1, 2, 20, 1e12)
    assert n_split == 1
    assert [j.name for j in work] == ["c:101-120", "c:121-140", "d"]
    assert [j.order for j in work] == [(0, 0), (0, 20), (1, 0)]
    assert [j.safe_name for j in work] == ["c_101-120", "c_121-140", "d"]
    work, too_big, _ = rec.plan_work(h, chrom, pos, 1, 2, 20, rec.phi_bytes(20))
    assert [j.name for j in too_big] == []  # each test needs phi_bytes(20)
    work, too_big, _ = rec.plan_work(h, chrom, pos, 1, 2, 20, rec.phi_bytes(20) - 1)
    assert work == [] and len(too_big) == 3
    work, _, _ = rec.plan_work(h, chrom, pos, 1, 21, 1000, 1e12)  # --min-snps
    assert [j.name for j in work] == ["c"]


def test_write_fasta(rec, tmp_path):
    """T-REC-7 (B21 j): A/C/N for 0/1/-1, one generic name per sample."""
    h = np.array([[0, 1, -1], [1, 0, 0]])
    rec.write_fasta(tmp_path / "x.fa", 3, h)
    assert (tmp_path / "x.fa").read_text() == ">s1\nAC\n>s2\nCA\n>s3\nNA\n"


def test_failed_phi_runs_exit_2_after_writing(rec, tmp_path, monkeypatch):
    """B6: Phi crashing on every contig is "not tested", exit status 2, and
    the table and summary are still written."""
    import stat
    import sys

    phi = tmp_path / "Phi"
    phi.write_text(f"#!{sys.executable}\nimport sys\nprint('Reading')\nsys.exit(3)\n")
    phi.chmod(phi.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(rec, "find_phi", lambda: str(phi))
    monkeypatch.setenv("POPGEN_LOG_DIR", str(tmp_path / "logs"))
    vcf = DATA.parents[1].parent / "examples" / "data" / "example.vcf"
    monkeypatch.setattr(sys, "argv", ["recombination", "--vcf", str(vcf),
                                      "--out", str(tmp_path / "rec")])
    with pytest.raises(SystemExit) as e:
        rec.main()
    assert e.value.code == 2
    res = tmp_path / "results"
    summary = (res / "rec.phi.summary.txt").read_text()
    assert "Phi runs failed: 2" in summary and "not tested" in summary
    table = (res / "rec.phi.contigs.tsv").read_text()
    assert table.count("Phi exit 3") == 2


def test_interpret_too_few_sites_for_permutation_is_not_a_failure(rec):
    """Phi exits 1 when even the permutation test has too few informative
    sites (real data: 3-4 informative sites); that is no failed run."""
    first = (DATA / "ctg2.log").read_text().split(rec.RERUN)[0]
    rerun = rec.PhiRun(1, "Found 4 informative sites.\n\nERROR: Too few informative "
                          "sites to test significance.  Try decreasing windowsize\n")
    r = rec.interpret(rec.PhiRun(0, first), rerun, 1000)
    assert r["p_phi_perm"] is None and not r["failed"]
    assert r["note"] == "too few informative sites for PHI"
