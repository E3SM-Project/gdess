import time, logging

_logger = logging.getLogger(__name__)


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


