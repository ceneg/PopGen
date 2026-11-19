#!/usr/bin/env python3
"""
Report the versions of the Python packages and external programs that each
script in scripts/ uses, for the methods section of a paper (printed, or
written to --out).
"""
import argparse
import ast
import importlib.metadata
import importlib.util
import logging
import platform
import sys
from pathlib import Path

from popgen.external import find_phi, phi_version
from popgen.io import atomic_write

log = logging.getLogger("citations")


def get_installed_version(module_name: str) -> str:
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


def scan_imports(path) -> set[str]:
    """
    Modules imported by a Python file: top-level names, plus the full dotted
    names of ``popgen.*`` modules.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except SyntaxError:
        log.warning(f"Warning: syntax error parsing {path}")
        return set()
    return imports_in_tree(tree)


def imports_in_tree(tree) -> set[str]:
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


def module_path(name: str) -> Path | None:
    """Source file of an importable module, or None."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    return Path(spec.origin) if spec and spec.origin else None


def scan_imports_recursive(module: str) -> set[str]:
    """
    Top-level third-party modules imported by ``module`` (a ``popgen.*``
    name), including those imported through other ``popgen`` modules.
    """
    seen, todo, found = set(), [module], set()
    stdlib = set(sys.stdlib_module_names)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        path = module_path(name)
        if path is None or path.suffix != ".py":
            continue
        for mod in scan_imports(path):
            if mod.startswith("popgen."):
                todo.append(mod)
            elif mod not in stdlib and mod != "popgen":
                found.add(mod)
    return found


def get_phi_version() -> str | None:
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


# External programs used by each analysis, with their version lookups
EXTERNAL_TOOLS = {"recombination": ["Phi"]}
VERSION_LOOKUPS = {"Phi": get_phi_version}


def analysis_modules() -> list[str]:
    """Names of the analysis modules (``popgen/analyses/*.py``), sorted."""
    here = Path(__file__).parent
    return sorted(p.stem for p in here.glob("*.py")
                  if p.stem not in ("__init__", "citations"))


def report() -> str:
    """The version report for every analysis (one section per script)."""
    python = f"Python {sys.version.split()[0]} ({platform.system()})"
    lines = [f"Software versions: {python}", "=" * 60, ""]
    for name in analysis_modules():
        imports = scan_imports_recursive(f"popgen.analyses.{name}")
        tools = EXTERNAL_TOOLS.get(name, [])
        if not imports and not tools:
            continue
        label = f"{name}.py"
        lines += [f"Script: {label}", "-" * (len(label) + 8)]
        if imports:
            lines.append("  [Python Modules]")
            for module in sorted(imports):
                lines.append(f"    - {module}: {get_installed_version(module)}")
        if tools:
            lines.append("  [External Software]")
            for tool in tools:
                info = VERSION_LOOKUPS[tool]()
                lines.append(f"    - {tool}: {info or 'Not found (WARNING)'}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the report to this file (default: print it; an existing "
        "file is overwritten)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    text = report()
    if args.out is None:
        print(text)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(args.out) as f:
        f.write(text + "\n")
    log.info(f"Report written to {args.out.absolute()}")


if __name__ == "__main__":
    main()
