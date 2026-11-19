"""Plotting helpers shared by the analyses."""

import matplotlib.pyplot as plt

from popgen.io import OutputPaths


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
