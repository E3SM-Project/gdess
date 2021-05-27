#!/usr/bin/env python
""" This is the command line interface for running co2 diagnostics

Example usage:
    >> ./bin/codima --help
    >> ./bin/codima trend --help
    >> ./bin/codima trend raw_data/noaa-obspack/nc/ --figure_savepath ./
    >> ./bin/codima seasonal --help
"""

# Generic/Built-in
import sys
from argparse import ArgumentParser

from co2_diag.recipes import seasonal_cycles, surface_trends
from co2_diag.recipes.seasonal_cycles import add_seasonal_cycle_args_to_parser
from co2_diag.recipes.surface_trends import add_surface_trends_args_to_parser


def main(args):
    """

    Parameters
    ----------
    recipe

    Returns
    -------

    """
    # Get the argument values. Then clear them from the namespace so the subcommands do not encounter them.
    verbosity = args.verbose
    del args.verbose
    recipe_name = args.subparser_name
    del args.subparser_name

    # Run the selected recipe
    if recipe_name == 'seasonal':
        seasonal_cycles(args, verbose=verbosity)
    elif recipe_name == 'trend':
        surface_trends(args, verbose=verbosity)

    return 0  # a clean, no-issue, exit


def parse_cli():
    """Input arguments are parsed.
    """
    parser = ArgumentParser(description="Generate diagnostic figures or metrics")
    parser.add_argument("--verbose", action='store_true')

    # Set up the argument parser for each recipe
    subparsers = parser.add_subparsers(title='available recipe subcommands', dest='subparser_name')  #, help='name of diagnostic recipe to run')
    #
    subparser_seasonal = subparsers.add_parser('seasonal', help='generate diagnostics of seasonal cycles')
    add_seasonal_cycle_args_to_parser(subparser_seasonal)
    #
    subparser_trend = subparsers.add_parser('trend', help='generate diagnostics of multidecadal trends')
    add_surface_trends_args_to_parser(subparser_trend)

    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_cli()

    sys.exit(main(parsed_args))
