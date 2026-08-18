"""Command line interface of ChI-SOM."""

import argparse
import sys

from typing import Optional, Sequence

from chisom.io import loading

EXIT_LOAD_ERROR = 1
EXIT_MISSING_GUI = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chisom",
        description="Fast self-organizing maps for cheminformatics.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the installed ChI-SOM version and exit",
    )
    subparsers = parser.add_subparsers(dest="command")

    view = subparsers.add_parser(
        "view",
        help="open the interactive viewer",
        description=(
            "Open the interactive viewer. Every artefact is optional and can also "
            "be loaded from the viewer's File menu."
        ),
    )
    view.add_argument(
        "-u",
        "--umatrix",
        metavar="PATH",
        help="U-matrix of the SOM, as a .npy file",
    )
    view.add_argument(
        "-b",
        "--bmus",
        metavar="PATH",
        help="BMU coordinates of the datapoints, as a .npy file",
    )
    view.add_argument(
        "-d",
        "--data",
        metavar="PATH",
        help=(
            "datapoint properties, as an HDF5 store (.h5, .hdf5), delimited text "
            "(.csv, .tsv, .txt) or Parquet (.parquet, .pq)"
        ),
    )
    view.add_argument(
        "--structure-column",
        metavar="NAME",
        help="dataset column holding the SMILES used to render structures",
    )
    view.add_argument(
        "--groups",
        metavar="GROUP",
        nargs="+",
        help="HDF5 only: the groups of the store to load, by default all of them",
    )
    view.add_argument(
        "--scaling-factor",
        type=int,
        default=3,
        metavar="N",
        help="interpolate the U-matrix by this factor for an anti-aliased view (default: 3)",
    )
    view.set_defaults(func=run_view)

    return parser


def run_view(args: argparse.Namespace) -> int:
    if args.groups and not args.data:
        print("chisom view: --groups needs --data to be given too.", file=sys.stderr)
        return EXIT_LOAD_ERROR

    try:
        umatrix = loading.load_umatrix(args.umatrix) if args.umatrix else None
        bmu_coordinates = loading.load_bmu_coordinates(args.bmus) if args.bmus else None
        data = (
            loading.load_dataset(args.data, group_subset=args.groups)
            if args.data
            else None
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"chisom view: {exc}", file=sys.stderr)
        return EXIT_LOAD_ERROR

    try:
        from chisom import start_chisom_viewer
    except ImportError as exc:
        print(f"chisom view: {exc}", file=sys.stderr)
        return EXIT_MISSING_GUI

    try:
        start_chisom_viewer(
            umatrix,
            bmu_coordinates,
            data,
            structure_info_column=args.structure_column,
            scaling_factor=args.scaling_factor,
        )
    except RuntimeError as exc:
        print(f"chisom view: {exc}", file=sys.stderr)
        return EXIT_LOAD_ERROR

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from chisom import __version__

        print(__version__)
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
