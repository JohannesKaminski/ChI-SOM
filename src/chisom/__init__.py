from importlib.metadata import version

from ._som import Som
from .io.plotting import plot_som

__all__ = [
    "Som",
    "plot_som",
    "start_chisom_viewer",
]


def __getattr__(name):
    if name == "start_chisom_viewer":
        from ._interface.gui import start_chisom_viewer

        return start_chisom_viewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = version("chi-som")
__author__ = "Johannes Kaminski"
__credits__ = "AG Koch"
