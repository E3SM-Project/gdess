from co2_diag.formatters.args import options_to_args, valid_year_string, valid_existing_path
from co2_diag.operations.time import year_to_datetime64
from typing import Union, Callable
import argparse, time, logging

_logger = logging.getLogger(__name__)


def add_shared_arguments_for_recipes(parser: argparse.ArgumentParser) -> None:
    """Add common recipe arguments to a parser object

    Parameters
    ----------
    parser
    """

    parser.add_argument('ref_data', nargs='?', default=None, type=valid_existing_path,
                        help='Filepath to the reference data folder')
    parser.add_argument('--start_yr', default="1958", type=valid_year_string,
                        help='Initial year cutoff. Default is 1958, which is the first year of the Mauna Loa CO2 record.')
    parser.add_argument('--end_yr', default="2014", type=valid_year_string,
                        help='Final year cutoff. Default is 2014, which is the final year for CMIP6 historical runs.')
    parser.add_argument('--figure_savepath', type=str, default=None, help='Filepath for saving generated figures')


def parse_recipe_options(options: Union[dict, argparse.Namespace],
                         recipe_specific_argument_adder: Callable[[argparse.ArgumentParser], None]
                         ) -> argparse.Namespace:
    """

    Parameters
    ----------
    options : Union[dict, argparse.Namespace]
        specifications for a given recipe execution
    recipe_specific_argument_adder : function
        a function that adds arguments defined for a particular recipe to a parser object

    Returns
    -------
    a parsed argument namespace
    """
    parser = argparse.ArgumentParser(description='Process surface observing station and CMIP data and compare. ')
    recipe_specific_argument_adder(parser)

    if isinstance(options, dict):
        # In this case, the options have not yet been parsed.
        params = options_to_args(options)
        if '--ref_data' in params:
            params.remove('--ref_data')  # remove this key because it is handled as a positional argument, not a kwarg.
        _logger.debug('Parameter argument string == %s', params)
        args = parser.parse_args(params)
    elif isinstance(options, argparse.Namespace):
        # In this case, the options have been parsed previously.
        _logger.debug('Parameters == %s', options)
        args = options
    else:
        raise TypeError('<%s> is an unexpected type of the recipe options', type(options))

    # Convert times to numpy.datetime64
    args.start_datetime = year_to_datetime64(args.start_yr)
    args.end_datetime = year_to_datetime64(args.end_yr)

    _logger.debug("Parsing is done.")
    return args


def benchmark_recipe(func):
    """A decorator for diagnostic recipe methods that provides timing info.
    """
    def display_time_and_call(*args, **kwargs):
        # Clock is started.
        start_time = time.time()
        # Recipe is run.
        returnval = func(*args, **kwargs)
        # Report the time this recipe took to execute.
        execution_time = (time.time() - start_time)
        _logger.info('recipe execution time (seconds): ' + str(execution_time))

        return returnval
    return display_time_and_call
