import argparse
import os
import shlex
from typing import Union


def options_to_args(options: dict) -> list:
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