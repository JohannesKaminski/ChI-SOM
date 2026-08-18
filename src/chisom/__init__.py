from importlib.metadata import version

from ._som import Som
from .analysis import u_distance
from .io.plotting import plot_som

__all__ = [
    "Som",
    "plot_som",
    "u_distance",
    "start_chisom_viewer",
]


def __getattr__(name):
    if name == "start_chisom_viewer":
        try:
            from ._interface.gui import start_chisom_viewer
        except ImportError as exc:
            raise ImportError(
                "The interactive viewer requires the optional GUI dependencies. "
                "Install them with `pip install 'chi-som[gui]'`"
            ) from exc

        return start_chisom_viewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = version("chi-som")
__author__ = "Johannes Kaminski"
__credits__ = "AG Koch"
