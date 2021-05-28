import os
import argparse
import shlex
import time
from typing import Union, Callable

from co2_diag.operations.time import ensure_dataset_datetime64, year_to_datetime64

import logging
_logger = logging.getLogger(__name__)


def options_to_args(options: dict) -> list[str]:
    """Convert a dictionary to a list of strings so that an ArgumentParser can parse it.

    Examples
    --------
    a = {'start_yr': "1980", 'end_yr': "2010"}
    >>> options_to_args(a)
    returns ['--start_yr', '1980', '--end_yr', '2010']
    """
    return shlex.split(' '.join([f"--{k} {v}" for k, v in options.items()]))


def is_some_none(val) -> bool:
    """Check if value is either a Python None object or the case-insensitive string 'None' """
    if val is None:
        return True
    elif isinstance(val, str) and (val.lower() == 'none'):
        return True
    else:
        return False


def nullable_int(val) -> Union[None, int]:
    """Validate whether a value's type is either an integer or none"""
    if is_some_none(val):
        return None
    if not isinstance(val, int):
        raise argparse.ArgumentTypeError("Value must be an integer, None, or 'None'.")
    return val


def nullable_str(val) -> Union[None, str]:
    """Validate whether a value's type is either a string or none"""
    if is_some_none(val):
        return None
    if not isinstance(val, str):
        raise argparse.ArgumentTypeError('Value must be an string or None.')
    return val


def valid_year_string(y) -> Union[None, str]:
    """Function used to validate 'year' argument passed in as a recipe option"""
    if is_some_none(y):
        return None
    elif isinstance(y, str) | isinstance(y, int):
        if 0 <= int(y) <= 10000:
            return str(y)
    raise argparse.ArgumentTypeError('Year must be a string or integer whose value is between 0 and 10,000.')


def valid_existing_path(p):
    """Function used to validate a file path argument passed in as a recipe option"""
    try:
        if os.path.exists(p):
            if os.access(p, os.R_OK):
                return p
    except TypeError:
        pass
    raise argparse.ArgumentTypeError('Path must exist and be readable.')


def add_shared_arguments_for_recipes(parser: argparse.PARSER) -> None:
    """Add common recipe arguments to a parser object

    Parameters
    ----------
    parser
    """
    parser.add_argument('ref_data', type=valid_existing_path, help='Filepath to the reference data folder')
    parser.add_argument('--start_yr', default="1960", type=valid_year_string, help='Initial year cutoff')
    parser.add_argument('--end_yr', default="2015", type=valid_year_string, help='Final year cutoff')
    parser.add_argument('--figure_savepath', type=str, default=None, help='Filepath for saving generated figures')


def parse_recipe_options(options: Union[dict, argparse.Namespace],
                         recipe_specific_argument_adder: Callable[[argparse.PARSER], None]
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
