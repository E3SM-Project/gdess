#!/usr/bin/env python
""" This is the command line interface for running co2 diagnostics

Example usage:
    >> ./bin/gdess_cli.py --help
    >> ./bin/gdess_cli.py trend --help
    >> ./bin/gdess_cli.py trend raw_data/noaa-obspack/nc/ --figure_savepath ./
    >> ./bin/gdess_cli.py seasonal --help
    >> ./bin/gdess_cli.py meridional --help
"""
from gdess.recipe_parsers import add_surface_trends_args_to_parser, add_seasonal_cycle_args_to_parser, \
    add_meridional_args_to_parser
from argparse import ArgumentParser
import sys


def main(args):
    # Get the argument values. Then clear them from the namespace so the subcommands do not encounter them.
    verbosity = args.verbose
    recipe_name = args.subparser_name
    del (args.verbose, args.subparser_name)

    # Run the selected recipe
    if recipe_name == 'trend':
        from gdess.recipes import surface_trends
        surface_trends(args, verbose=verbosity)

    elif recipe_name == 'seasonal':
        from gdess.recipes import seasonal_cycles
        seasonal_cycles(args, verbose=verbosity)

    elif recipe_name == 'meridional':
        from gdess.recipes import meridional_gradient
        meridional_gradient(args, verbose=verbosity)

    return 0  # a clean, no-issue, exit


def parse_cli():
    """Input arguments are parsed.
    """
    parser = ArgumentParser(description="Run a CO2 diagnostic recipe to generate figures and metrics",
                            fromfile_prefix_chars='@')
    parser.add_argument("--verbose", action='store_true')

    # Redefine parser function to allow multiple arguments (i.e., nargs='*') on a single line when read from a file
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # A separate subparser is set up for each recipe to handle its specific input arguments.
    subparsers = parser.add_subparsers(title='available recipe subcommands', dest='subparser_name')  #, help='name of diagnostic recipe to run')
    #
    subparser_trend = subparsers.add_parser('trend', help='generate diagnostics of multidecadal trends')
    add_surface_trends_args_to_parser(subparser_trend)
    #
    subparser_seasonal = subparsers.add_parser('seasonal', help='generate diagnostics of seasonal cycles')
    add_seasonal_cycle_args_to_parser(subparser_seasonal)
    #
    subparser_meridional = subparsers.add_parser('meridional', help='generate diagnostics of meridional gradient')
    add_meridional_args_to_parser(subparser_meridional)

    # Print the help message if no arguments are supplied at the command line.
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main(parse_cli()))
