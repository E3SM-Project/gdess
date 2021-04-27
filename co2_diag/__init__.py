import logging
from typing import Union


def set_verbose(logger,
                verbose: Union[bool, str] = False
                ) -> None:
    """
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

    logconfig_path = os.path.dirname(os.path.realpath(__file__)) + '/config/log_config.json'
    with open(logconfig_path, 'r') as logging_configuration_file:
        config_dict = json.load(logging_configuration_file)

    logging.config.dictConfig(config_dict)


def _change_log_level(a_logger, level):
    a_logger.setLevel(level)
    for handler in a_logger.handlers:
        handler.setLevel(level)


_config_logger()
