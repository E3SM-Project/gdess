import argparse
import shlex
import time
from typing import Union

import logging
_logger = logging.getLogger(__name__)


def options_to_args(options: dict):
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
    elif isinstance(val, str):
        if val.lower() == 'none':
            return True
    else:
        return False


def nullable_int(val) -> Union[None, int]:
    """Validate whether a value's type is either an integer or none"""
    if is_some_none(val):
        return None
    if not isinstance(val, int):
        raise argparse.ArgumentTypeError('Value must be an integer')
    return val


def nullable_str(val) -> Union[None, str]:
    """Validate whether a value's type is either a string or none"""
    if is_some_none(val):
        return None
    if not isinstance(val, str):
        raise argparse.ArgumentTypeError('Value must be an string')
    return val


def valid_year_string(y) -> Union[None, str]:
    """Function used to validate 'year' argument passed in as a recipe option"""
    if is_some_none(y):
        return None
    elif isinstance(y, str) | isinstance(y, int):
        if 0 <= int(y) <= 10000:
            return str(y)
    raise argparse.ArgumentTypeError('Year must be a string or integer whose value is between 0 and 10,000.')


def benchmark_recipe(func):
    """A decorator for diagnostic recipe methods that provides timing info.

    This is used to reduce code duplication.
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
