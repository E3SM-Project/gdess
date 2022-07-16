import os
import json
import time
import logging
import configparser
import pkg_resources
from typing import Union, Callable

_logger = logging.getLogger(__name__)

def load_config_file() -> configparser.ConfigParser:
    """Read the package configuration file

    Returns
    -------
    configparser.ConfigParser
    """
    config = configparser.ConfigParser(os.environ,
                                       interpolation=configparser.ExtendedInterpolation(),
                                       comment_prefixes=('#', ';'))

    path = 'config/defaults.ini'  # always use slash
    filepath = pkg_resources.resource_filename(__package__, path)
    config.read(filepath)

    return config


def load_stations_dict() -> dict:
    """Get the dictionary of stations

    Returns
    -------
    dict
    """
    path = 'config/stations_dict.json'  # always use slash
    filepath = pkg_resources.resource_filename(__package__, path)
    with open(filepath, 'r') as f:
        result = json.load(f)

    return result


def set_verbose(logger: logging.Logger,
                verbose: Union[bool, str] = False
                ) -> None:
    """Set the level of the passed logger

    Parameters
    ----------
    logger: logging.Logger
    verbose: Union[bool, str]
        can be either True, False, or a string for level such as "INFO, DEBUG, etc."
    """
    logger.setLevel(validate_verbose(verbose))


def validate_verbose(verbose: Union[bool, str] = False) -> Union[int, str]:
    """Convert a verbosity argument to a logging level

    Parameters
    ----------
    verbose
        either True, False, or a string for level such as "INFO, DEBUG, etc."

    Returns
    -------
    int or str
        A logging verbosity level or string that corresponds to a verbosity level

    """
    if verbose is True:
        level_to_set = logging.DEBUG
    elif verbose is not None:
        level_to_set = verbose
    elif verbose is None:
        level_to_set = logging.WARN
    else:
        raise ValueError("Unexpected/unhandled verbose option <%s>. "
                         "Please use True, False or a string for level such as 'INFO, DEBUG, etc.'", verbose)
    return level_to_set


def _config_logger():
    """Configure the root logger"""
    import os
    import logging.config
    import json

    logconfig_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'config',
                                  'log_config.json')
    with open(logconfig_path, 'r') as logging_configuration_file:
        config_dict = json.load(logging_configuration_file)

    logging.config.dictConfig(config_dict)


def _change_log_level(a_logger: logging.Logger,
                      level: int):
    a_logger.setLevel(level)
    for handler in a_logger.handlers:
        handler.setLevel(level)


def benchmark_recipe(func: Callable):
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


_config_logger()
