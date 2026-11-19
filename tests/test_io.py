import gzip

import numpy as np
import pytest

from popgen.io import (
    atomic_write,
    ensure_output_dirs,
    load_metadata,
    read_vcf,
    safe_name,
    safe_names,
    write_table,
)

VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
)


def test_read_vcf_rejects_multiallelic(tmp_path):
    vcf = tmp_path / "multi.vcf"
    vcf.write_text(VCF_HEADER + "c1\t5\t.\tA\tC,G\t.\tPASS\t.\tGT\t1\t2\n")
    with pytest.raises(SystemExit):
        read_vcf(str(vcf), fields=["samples", "calldata/GT"])


def test_read_vcf_biallelic(tmp_path):
    vcf = tmp_path / "ok.vcf"
    vcf.write_text(VCF_HEADER + "c1\t5\t.\tA\tC\t.\tPASS\t.\tGT\t0\t1\n")
    samples, gt, _ = read_vcf(str(vcf), fields=["samples", "calldata/GT"])
    assert list(samples) == ["A", "B"]
    assert gt.shape[:2] == (1, 2)


def test_read_vcf_numalt_added_and_samples_only_read(tmp_path):
    vcf = tmp_path / "multi.vcf"
    vcf.write_text(VCF_HEADER + "c1\t5\t.\tA\tC,G\t.\tPASS\t.\tGT\t1\t2\n")
    with pytest.raises(SystemExit):
        read_vcf(str(vcf), fields=["samples", "calldata/GT", "variants/POS"])
    samples, gt, _ = read_vcf(str(vcf), fields=["samples"])  # header only
    assert list(samples) == ["A", "B"] and gt is None


def test_read_vcf_empty_or_missing_exits(tmp_path):
    header_only = tmp_path / "empty.vcf"
    header_only.write_text(VCF_HEADER)
    with pytest.raises(SystemExit):
        read_vcf(str(header_only), fields=["samples", "calldata/GT"])
    with pytest.raises(SystemExit):
        read_vcf(str(tmp_path / "missing.vcf"), fields=["samples", "calldata/GT"])


def test_read_vcf_gzip_same_as_plain(tmp_path):
    text = (VCF_HEADER + "c1\t5\t.\tA\tC\t.\tPASS\t.\tGT\t0\t1\n"
            + "c1\t9\t.\tG\tT\t.\tPASS\t.\tGT\t1\t.\n")
    plain = tmp_path / "v.vcf"
    plain.write_text(text)
    packed = tmp_path / "v.vcf.gz"
    with gzip.open(packed, "wt") as f:
        f.write(text)
    fields = ["samples", "calldata/GT", "variants/POS"]
    s1, g1, c1 = read_vcf(str(plain), fields=fields)
    s2, g2, c2 = read_vcf(str(packed), fields=fields)
    assert list(s1) == list(s2)
    np.testing.assert_array_equal(np.asarray(g1), np.asarray(g2))
    np.testing.assert_array_equal(c1["variants/POS"], c2["variants/POS"])


@pytest.mark.filterwarnings("ignore:'GT' FORMAT header not found")
def test_read_vcf_without_genotypes_exits(tmp_path):
    """T-VCF-2 (B21 d): no FORMAT/GT columns."""
    vcf = tmp_path / "sites.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\t"
                   "INFO\nc1\t5\t.\tA\tC\t.\tPASS\t.\n")
    with pytest.raises(SystemExit):
        read_vcf(str(vcf), fields=["samples", "calldata/GT"])


# --- Output paths and atomic writes ---

def test_ensure_output_dirs(tmp_path):
    paths = ensure_output_dirs(str(tmp_path / "run"))
    assert (tmp_path / "results" / "png").is_dir()
    assert (tmp_path / "results" / "svg").is_dir()
    assert paths.table("tsv") == tmp_path / "results" / "run.tsv"
    assert paths.png("pca") == tmp_path / "results" / "png" / "run.pca.png"
    assert paths.svg("pca") == tmp_path / "results" / "svg" / "run.pca.svg"


@pytest.mark.parametrize("text, expected", [
    ("host/substrate", "host_substrate"),
    ("dN/dS", "dN_dS"),
    ("lineage", "lineage"),
    ("a b  c", "a_b_c"),
    ("/", "_"),
])
def test_safe_name(text, expected):
    """T-OUT-1 (B11)."""
    assert safe_name(text) == expected


def test_safe_names_clash_exits():
    assert safe_names(["a/b", "c"]) == {"a/b": "a_b", "c": "c"}
    with pytest.raises(SystemExit):
        safe_names(["a/b", "a b"])


def test_atomic_write_error_leaves_nothing(tmp_path):
    """T-OUT-2."""
    new = tmp_path / "new.tsv"
    with pytest.raises(RuntimeError):
        with atomic_write(new) as f:
            f.write("partial")
            raise RuntimeError("killed")
    assert not new.exists()
    old = tmp_path / "old.tsv"
    old.write_text("kept")
    with pytest.raises(RuntimeError):
        with atomic_write(old) as f:
            f.write("partial")
            raise RuntimeError("killed")
    assert old.read_text() == "kept"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["old.tsv"]


def test_write_table_matches_to_csv(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan]}, index=["x", "y"])
    df.to_csv(tmp_path / "plain.tsv", sep="\t")
    write_table(df, tmp_path / "atomic.tsv", sep="\t")
    plain, atomic = tmp_path / "plain.tsv", tmp_path / "atomic.tsv"
    assert plain.read_bytes() == atomic.read_bytes()


# --- Metadata (B1) ---

def write_meta(tmp_path, text):
    path = tmp_path / "meta.tsv"
    path.write_text(text)
    return str(path)


def test_load_metadata_follows_vcf_order(tmp_path):
    """T-META-6."""
    meta = write_meta(tmp_path, "strain\tg\nB\tx\nZ\tz\nA\ty\n")
    out = load_metadata(meta, ["A", "B"])
    assert out.index.name == "strain"
    assert list(out.index) == ["A", "B"]  # VCF order; surplus Z dropped
    assert list(out["g"]) == ["y", "x"]


def test_load_metadata_numeric_names_stay_strings(tmp_path):
    """T-META-1."""
    meta = write_meta(tmp_path, "strain\tyear\n101\t2019\n102\t2020\n")
    out = load_metadata(meta, np.array(["101", "102"], dtype=object))
    assert list(out.index) == ["101", "102"]
    assert list(out["year"]) == ["2019", "2020"]


def test_load_metadata_leading_zeros_and_whitespace(tmp_path):
    meta = write_meta(tmp_path, " strain \tg \n007 \tx\n 8\ty\n")
    out = load_metadata(meta, ["007", "8"])
    assert list(out.columns) == ["g"]
    assert list(out["g"]) == ["x", "y"]


def test_load_metadata_no_overlap_exits(tmp_path, caplog):
    """T-META-2."""
    meta = write_meta(tmp_path, "strain\tyear\n101\t2019\n102\t2020\n")
    with pytest.raises(SystemExit):
        load_metadata(meta, ["X", "Y"])
    assert "X, Y" in caplog.text and "101, 102" in caplog.text


@pytest.mark.parametrize("text", [
    "strain\tg\nA\tx\nA\ty\nB\tz\n",  # duplicate (T-META-3)
    "strain\tg\nA\tx\n\ty\n",  # empty strain
    "id\tg\nA\tx\n",  # no strain column
])
def test_load_metadata_invalid_exits(tmp_path, text):
    with pytest.raises(SystemExit):
        load_metadata(write_meta(tmp_path, text), ["A", "B"])


def test_load_metadata_unreadable_exits(tmp_path):
    with pytest.raises(SystemExit):
        load_metadata(tmp_path / "missing.tsv", ["A"])


def test_load_metadata_strain_only(tmp_path):
    """T-META-4."""
    meta = write_meta(tmp_path, "strain\nA\nB\n")
    out = load_metadata(meta, ["A", "B"])
    assert out.shape == (2, 0)
    assert list(out.index) == ["A", "B"]


def test_load_metadata_keeps_samples_without_metadata(tmp_path, caplog):
    """T-META-5."""
    meta = write_meta(tmp_path, "strain\tg\nA\tx\nB\ty\n")
    with caplog.at_level("WARNING"):
        out = load_metadata(meta, ["A", "B", "C"])
    assert list(out.index) == ["A", "B", "C"]
    assert out.loc["C"].isna().all()
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
