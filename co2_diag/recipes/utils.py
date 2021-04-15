import shlex
import time

import logging
_logger = logging.getLogger(__name__)


def get_recipe_param(param_dict,
                     param_key: str,
                     default_value=None,
                     type=None):
    """Validate a parameter in the parameter dictionary, and return default if it is not in the dictionary.

    Parameters
    ----------
    param_dict
    param_key
    default_value

    Returns
    -------
    The value from the dictionary, which can be of any type
    """
    value = default_value
    if param_dict and (param_key in param_dict):
        value = param_dict[param_key]
        if not not type:
            if not isinstance(value, type):
                raise TypeError(f'{param_key} param should have type {type}. It has type <{type(value)}>.')
    return value


def options_to_args(options: dict):
    """Convert a dictionary to a list of strings so that an ArgumentParser can parse it.

    Examples
    --------
    a = {'start_yr': "1980", 'end_yr': "2010"}
    >>> options_to_args(a)
    returns ['--start_yr', '1980', '--end_yr', '2010']
    """
    return shlex.split(' '.join([f"--{k} {v}" for k, v in options.items()]))


def valid_year_string(y):
    """Function used to validate 'year' argument passed in as a recipe option"""
    if y:
        if isinstance(y, str) | isinstance(y, int):
            if 0 <= int(y) <= 10000:
                return str(y)
    raise TypeError('Year must be a string or integer whose value is between 0 and 10,000.')


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
