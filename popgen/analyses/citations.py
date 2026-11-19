#!/usr/bin/env python3
"""
Report the versions of the Python packages and external programs that each
script in scripts/ uses, for the methods section of a paper.
"""
import argparse
import ast
import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from popgen.external import find_phi, phi_version


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
        m for m in modules if m.startswith("popgen.")
    }


def scan_imports_recursive(script_path, project_root):
    """
    Top-level third-party modules imported by a script, including those
    imported through the project's own ``popgen`` package.
    """
    seen, todo, found = set(), [script_path], set()
    stdlib = get_stdlib_modules()
    while todo:
        path = todo.pop()
        if path in seen or not Path(path).exists():
            continue
        seen.add(path)
        for mod in scan_imports(path):
            if mod.startswith("popgen."):
                todo.append(str(Path(project_root, *mod.split("."))) + ".py")
            elif mod not in stdlib and mod not in ("popgen", "scripts"):
                found.add(mod)
    return found


def get_phi_version():
    """
    PhiPack version: Phi has no version option, so it is read from the file
    that tools/install_phipack.py writes next to Phi, or from conda.
    """
    path = find_phi()
    if not path:
        return None
    v = phi_version(path)
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
        default=str(Path("results", "citations.txt")),
        help="Report file (default: results/citations.txt in the current "
        "directory; an existing file is overwritten)",
    )
    args = parser.parse_args()

    # Determine project root based on script location (for finding scripts)
    scripts_dir = Path(__file__).absolute().parent

    out_file = Path(args.out).absolute()
    out_file.parent.mkdir(parents=True, exist_ok=True)

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

    script_files = sorted(str(p) for p in scripts_dir.glob("*.py"))

    if not script_files:
        print(f"Warning: No scripts found in {scripts_dir}")

    for script_path in script_files:
        val_name = Path(script_path).name
        if val_name in ("__init__.py", Path(__file__).name):
            continue

        external_imports = scan_imports_recursive(
            script_path, str(scripts_dir.parents[1])
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
