import os
import pandas as pd


def validate_metadata(meta_path, strict=True):
    """
    Validate metadata file structure and content.
    Returns True if valid, raises error or returns False if invalid.
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    try:
        df = pd.read_csv(meta_path, sep="\t")
    except Exception as e:
        raise ValueError(f"Failed to read metadata: {e}")

    if "strain" not in df.columns:
        raise ValueError("Metadata must contain 'strain' column.")

    if df.empty:
        raise ValueError("Metadata file is empty.")

    if df["strain"].isnull().any():
        raise ValueError("Metadata contains null values in 'strain' column.")

    if df["strain"].duplicated().any():
        if strict:
            raise ValueError("Metadata contains duplicate strain names.")

    return True


def validate_vcf(vcf_path):
    """
    Validate that the VCF file exists and is not empty (its content is checked
    when ``src.utils.read_vcf`` parses it).
    """
    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    if os.path.getsize(vcf_path) == 0:
        raise ValueError("VCF file is empty.")

    return True


def validate_inputs(args):
    """
    Validation wrapper for script arguments.
    """
    if hasattr(args, "meta") and args.meta:
        validate_metadata(args.meta)

    if hasattr(args, "vcf") and args.vcf:
        validate_vcf(args.vcf)
