import argparse, logging, os, shlex, tempfile
from typing import Union

_logger = logging.getLogger(__name__)


def options_to_args(options: dict) -> list:
    """Convert a dictionary to a list of strings so that an ArgumentParser can parse it.

    Parameters
    ----------
    options : dict

    Examples
    --------
    a = {'start_yr': "1980", 'end_yr': "2010"}
    >>> options_to_args(a)
    returns ['--start_yr', '1980', '--end_yr', '2010']

    Returns
    -------
    list
    """
    return shlex.split(' '.join([f"--{k} {v}" for k, v in options.items()]))


def is_some_none(val) -> bool:
    """Check if value is either a Python None object or the case-insensitive string 'None'
    """
    if val is None:
        return True
    elif isinstance(val, str) and (val.lower() == 'none'):
        return True
    else:
        return False


def nullable_int(val: Union[None, int]) -> Union[None, int]:
    """Validate whether a value's type is either an integer or none

    Parameters
    ----------
    val
        value to validate

    Raises
    ------
    argparse.ArgumentTypeError
        if the value is not either an integer or None

    Returns
    -------
    int or None
    """
    if is_some_none(val):
        return None
    if not isinstance(val, int):
        raise argparse.ArgumentTypeError("Value must be an integer, None, or 'None'.")
    return val


def nullable_str(val: Union[None, str]) -> Union[None, str]:
    """Validate whether a value's type is either a string or none

    Parameters
    ----------
    val
        value to validate

    Raises
    ------
    argparse.ArgumentTypeError
        if the value is not either a string or None

    Returns
    -------
    str or None
    """
    if is_some_none(val):
        return None
    if not isinstance(val, str):
        raise argparse.ArgumentTypeError('Value must be an string or None.')
    return val


def valid_year_string(y: Union[str, int]) -> Union[None, str]:
    """Validate 'year' argument passed in as a recipe option

    Parameters
    ----------
    y
        year to validate

    Raises
    ------
    argparse.ArgumentTypeError
        if the year is not a valid string or positive integer less than 10,000

    Returns
    -------
    str or None
    """
    if is_some_none(y):
        return None
    elif isinstance(y, str) | isinstance(y, int):
        if 0 <= int(y) <= 10000:
            return str(y)
    raise argparse.ArgumentTypeError('Year must be a string or integer whose value is between 0 and 10,000.')


def valid_existing_path(p: Union[str, os.PathLike]
                        ) -> Union[str, os.PathLike]:
    """Validate a filepath argument passed in as a recipe option

    Parameters
    ----------
    p
        path to validate

    Raises
    ------
    argparse.ArgumentTypeError
        if the input path does not exist and/or is not writable

    Returns
    -------
    str or os.PathLike
    """
    try:
        if os.path.exists(p):
            if os.access(p, os.R_OK):
                return p
    except TypeError:
        pass
    raise argparse.ArgumentTypeError('Path must exist and be readable. <%s> is not.' % p)


def valid_writable_path(p: Union[str, os.PathLike]
                        ) -> Union[bool, str, os.PathLike]:
    """Validate a filepath argument passed in as a recipe option

    Parameters
    ----------
    p
        path to validate

    Raises
    ------
    argparse.ArgumentTypeError
        if the input path is not valid and writable

    Returns
    -------
    bool, str, or os.PathLike
    """
    def canmakeit():
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix='_temp', dir=os.path.dirname(p)) as file_object:
                _logger.debug("Testing - successfully created temporary file (%s)." % file_object.name)
        except:
            raise argparse.ArgumentTypeError('Path must be valid and writable. <%s> is not.' % p)
        return True

    if (p is None) or (not canmakeit()):
        raise argparse.ArgumentTypeError('Path must be valid and writable. <%s> is not.' % p)

    return p
