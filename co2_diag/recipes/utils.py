import time

from co2_diag.dataset_operations.multiset import _multiset_logger


def get_recipe_param(param_dict, param_key: str, default_value=None):
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
    return value


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
        _multiset_logger.info('recipe execution time (seconds): ' + str(execution_time))

        return returnval
    return display_time_and_call
