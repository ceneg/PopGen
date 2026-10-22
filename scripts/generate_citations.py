#!/usr/bin/env python3
"""
Report the versions of the Python packages and external programs that each
script in scripts/ uses, for the methods section of a paper.
"""
import argparse
import ast
import glob
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import find_phi

# Written by tools/install_phipack.py next to the Phi program it builds
PHI_VERSION_FILE = "PhiPack.version"


def get_stdlib_modules():
    """
    Return the set of standard library module names.
    """
    return set(sys.stdlib_module_names)


def get_installed_version(module_name):
    """
    Try to get version of an installed package.
    Handles mapping from import name to distribution name
    (e.g. sklearn -> scikit-learn).
    """
    # Common mappings
    mapping = {
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "allel": "scikit-allel",
        "Bio": "biopython",
        "dns": "dnspython",
        "dateutil": "python-dateutil",
    }

    dist_name = mapping.get(module_name, module_name)

    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        # Try to import and check __version__ as fallback
        try:
            mod = __import__(module_name)
            if hasattr(mod, "__version__"):
                return mod.__version__
        except ImportError:
            pass
        return "Not installed"


def scan_imports(script_path):
    """
    Parse a python script and return a set of top-level imported module names.
    """
    with open(script_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            print(f"Warning: Syntax error parsing {script_path}")
            return set()

    return imports_in_tree(tree)


def imports_in_tree(tree):
    """
    Return the dotted module names imported in an AST.
    """
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
    return {m.split(".")[0] for m in modules} | {
        m for m in modules if m.startswith("src.")
    }


def scan_imports_recursive(script_path, project_root):
    """
    Top-level third-party modules imported by a script, including those
    imported through the project's own ``src`` package.
    """
    seen, todo, found = set(), [script_path], set()
    stdlib = get_stdlib_modules()
    while todo:
        path = todo.pop()
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        for mod in scan_imports(path):
            if mod.startswith("src."):
                todo.append(os.path.join(project_root, *mod.split(".")) + ".py")
            elif mod not in stdlib and mod not in ("src", "scripts"):
                found.add(mod)
    return found


def get_conda_version(package_name):
    """
    Try to look up version in conda list.
    """
    if not shutil.which("conda"):
        return None

    try:
        # Check specific package: conda list -p PREFIX package_name
        # use sys.prefix to dynamically target the current active environment
        # This is robust and not hardcoded.
        result = subprocess.run(
            ["conda", "list", "-p", sys.prefix, package_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Parse output:
        # # packages in environment at ...
        # #
        # # Name                    Version                   Build  Channel
        # phipack                   1.1                  h7b50bb2_6    bioconda

        for line in result.stdout.splitlines():
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == package_name:
                return parts[1]

    except Exception:
        pass
    return None


def get_phi_version():
    """
    PhiPack version: Phi has no version option, so it is read from the file
    that tools/install_phipack.py writes next to Phi, or from conda.
    """
    path = find_phi()
    if not path:
        return None
    version_file = Path(path).with_name(PHI_VERSION_FILE)
    if version_file.is_file():
        return f"{version_file.read_text().strip()} (found at {path})"
    v = get_conda_version("phipack")
    if v:
        return f"{v} (found at {path})"
    return f"Detected at {path} (version unknown)"


def get_external_version(tool_name):
    """
    Try to get version of external tool.
    """
    if tool_name == "Phi":
        return get_phi_version()

    path = shutil.which(tool_name)
    if not path:
        return None

    # Try common version flags
    for flag in ["--version", "-V", "-v", "version"]:
        try:
            result = subprocess.run(
                [tool_name, flag], capture_output=True, text=True, timeout=2
            )
            output = result.stdout + result.stderr

            # Simple regex for x.y.z
            match = re.search(r"(\d+\.\d+(\.\d+)?)", output)
            if match:
                return f"{match.group(1)} (found at {path})"
        except Exception:
            pass

    return f"Detected at {path} (Version finding failed)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--out",
        default=os.path.join("results", "citations.txt"),
        help="Report file (default: results/citations.txt in the current "
        "directory; an existing file is overwritten)",
    )
    args = parser.parse_args()

    # Determine project root based on script location (for finding scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = script_dir

    out_file = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Map scripts to external tools
    external_tools_map = {
        "recombination.py": ["Phi"],
    }

    report_lines = []
    report_lines.append(
        f"Software versions: Python {sys.version.split()[0]} ({platform.system()})"
    )
    report_lines.append("=" * 60)
    report_lines.append("")

    script_files = sorted(glob.glob(os.path.join(scripts_dir, "*.py")))

    if not script_files:
        print(f"Warning: No scripts found in {scripts_dir}")

    for script_path in script_files:
        val_name = os.path.basename(script_path)
        if val_name in ("__init__.py", os.path.basename(__file__)):
            continue

        external_imports = scan_imports_recursive(
            script_path, os.path.dirname(scripts_dir)
        )

        # Check for external tools
        tools = external_tools_map.get(val_name, [])

        if not external_imports and not tools:
            continue

        report_lines.append(f"Script: {val_name}")
        report_lines.append("-" * (len(val_name) + 8))

        # Python Modules
        if external_imports:
            report_lines.append("  [Python Modules]")
            for module in sorted(external_imports):
                version = get_installed_version(module)
                report_lines.append(f"    - {module}: {version}")

        # External Software
        if tools:
            report_lines.append("  [External Software]")
            for tool in tools:
                ver_info = get_external_version(tool)
                if ver_info:
                    report_lines.append(f"    - {tool}: {ver_info}")
                else:
                    report_lines.append(f"    - {tool}: Not found (WARNING)")

        report_lines.append("")

    with open(out_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report gathered in {out_file}")
    # Print content
    with open(out_file, "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
