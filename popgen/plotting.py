"""Plotting helpers shared by the analyses: figure saving and category colours."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from popgen.io import OutputPaths

UNKNOWN = "unknown"
UNKNOWN_COLOR = (0.5, 0.5, 0.5)


def save_figure(fig, paths: OutputPaths, stem: str, **savefig_kwargs) -> None:
    """
    Save ``fig`` as ``png/NAME.<stem>.png`` and ``svg/NAME.<stem>.svg``, then
    close it.

    Args:
        fig: Matplotlib figure.
        paths: From ``ensure_output_dirs``.
        stem: File-name part after the output name.
        savefig_kwargs: Passed to ``savefig`` for both files; a ``png_kwargs``
            dict holds arguments for the PNG only (e.g. ``dpi``).
    """
    png_kwargs = savefig_kwargs.pop("png_kwargs", {})
    fig.savefig(paths.png(stem), **savefig_kwargs, **png_kwargs)
    fig.savefig(paths.svg(stem), **savefig_kwargs)
    plt.close(fig)


def categories(values) -> pd.Series:
    """
    Metadata values as category labels: strings, with missing values as
    ``UNKNOWN``.

    Args:
        values: A metadata column (Series) or any sequence.

    Returns:
        pd.Series: Labels, same index as ``values`` if it is a Series.
    """
    s = values
    if not isinstance(s, pd.Series):
        s = pd.Series(list(values), dtype=object)
    return s.astype(object).where(s.notna(), UNKNOWN).astype(str)


def category_colors(values) -> dict[str, tuple]:
    """
    One colour per category, the same in every figure for the same values.

    Non-missing values are sorted as strings and coloured from ``tab20``
    (up to 20 categories) or ``hsv`` (more); ``UNKNOWN`` (grey) is added last
    only if a value is missing.

    Args:
        values: Metadata values (missing = None/NaN).

    Returns:
        dict: {category label: RGB tuple}.
    """
    s = pd.Series(list(values), dtype=object)
    known = sorted({str(v) for v in s[s.notna()]})
    known = [v for v in known if v != UNKNOWN]
    palette = "tab20" if len(known) <= 20 else "hsv"
    colors = dict(zip(known, sns.color_palette(palette, len(known))))
    if s.isna().any() or UNKNOWN in set(s[s.notna()].astype(str)):
        colors[UNKNOWN] = UNKNOWN_COLOR
    return colors


def add_category_legend(target, colors: dict, title: str):
    """
    Legend of category colours beside an Axes or Figure (centre left of its
    right edge).

    Args:
        target: Matplotlib Axes or Figure.
        colors: {label: colour} from ``category_colors``.
        title: Legend title.

    Returns:
        The Legend.
    """
    handles = [Patch(facecolor=c, edgecolor="none", label=label)
               for label, c in colors.items()]
    return target.legend(handles=handles, title=title, loc="center left",
                         bbox_to_anchor=(1.0, 0.5), frameon=False)
