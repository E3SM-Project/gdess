import time
import pickle
from typing import Union

from co2_diag.dataset_operations.datasetdict import DatasetDict

import logging
_multiset_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))


class Multiset:
    """Useful class for working simultaneously with multiple, consistent xarray Datasets

    """
    def __init__(self, verbose=False):
        """
        This class is a template against which we can run recipes, with an order of operations:
            - Step A: datasets loaded, in their original form
            - Step B: datasets that have been preprocessed
            - Step C: datasets that have operations lazily queued
            - Step D: datasets that have been fully processed by executing operations on them

        Parameters
        ----------
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        self.stepA_original_datasets: Union[DatasetDict, None] = None
        self.stepB_preprocessed_datasets: Union[DatasetDict, None] = None
        self.stepC_prepped_for_execution_datasets: DatasetDict = DatasetDict(dict())
        self.stepD_latest_executed_datasets: DatasetDict = DatasetDict(dict())

        self._set_multiset_verbose(verbose)

    # def datasets_to_file(self, filename: str = 'cmip_collection.latest_executed_datasets.pickle',):
    #     """Pickle the latest executed dataset dictionary using the highest protocol available.
    #
    #     Parameters
    #     ----------
    #     filename
    #
    #     """
    #     with open(filename, 'wb') as f:
    #         pickle.dump(self.latest_executed_datasets, f, pickle.HIGHEST_PROTOCOL)

    def datasets_from_file(self,
                           filename: str = 'cmip_collection.latest_executed_datasets.pickle',
                           replace: bool = False):
        """Load a dataset dictionary from a saved pickle file.

        Parameters
        ----------
        filename
        replace

        Returns
        -------
            None
        """
        with open(filename, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            le_datasets = pickle.load(f)

        if replace:
            self.stepD_latest_executed_datasets = le_datasets
        else:
            pass

    def __repr__(self):
        obj_attributes = sorted([k for k in self.__dict__.keys()
                                 if not self.stepC_prepped_for_execution_datasets[k].startswith('_')])

        # String representation is built.
        strrep = f"Multiset: \n" + \
                 self._original_datasets_list_str() + \
                 f"\n" \
                 f"\t all attributes:%s" % '\n\t\t\t'.join(obj_attributes)

        return strrep

    def _original_datasets_list_str(self):
        if self.stepA_original_datasets:
            return '\n\t'.join(self.stepA_original_datasets.keys())
        else:
            return ''

    def _set_multiset_verbose(self, verbose: Union[bool, str] = False):
        # verbose can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        _multiset_logger.setLevel(self._validate_verbose(verbose))

    @staticmethod
    def _validate_verbose(verbose: Union[bool, str] = False):
        # verbose can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        if verbose is True:
            level_to_set = logging.DEBUG
        elif verbose is not None:
            level_to_set = verbose
        elif verbose is None:
            level_to_set = logging.WARN
        else:
            raise ValueError("Unexpect/unhandled verbose option <%s>. "
                             "Please use True, False or a string for level such as 'INFO, DEBUG, etc.'", verbose)
        return level_to_set


def run_recipe(func):
    """A decorator for diagnostic recipe methods that provides timing info. Reduces code duplication.
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
