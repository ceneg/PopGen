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


@pytest.mark.xfail(strict=True, reason="B6")
def test_holm_refuses_nan(rec):
    # v1.01: [0.004, 1.0, 1.0, 1.0], one NaN erases every later rank
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
