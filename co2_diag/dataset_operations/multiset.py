import time
import pickle
from typing import Union

from co2_diag.dataset_operations.datasetdict import DatasetDict

import logging
_multiset_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))


class Multiset:
    """Useful class for working simultaneously with multiple, consistent xarray Datasets."""

    def __init__(self, verbose=False):
        """
        This class is a template against which we can run recipes, with an order of operations:
            - Step A: datasets loaded, in their original form
            - Step B: datasets that have been preprocessed
            - Step C: datasets that have operations lazily queued or fully processed

        Parameters
        ----------
        verbose
            either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        self.stepA_original_datasets: Union[DatasetDict, None] = None
        self.stepB_preprocessed_datasets: Union[DatasetDict, None] = None
        self.stepC_prepped_datasets: DatasetDict = DatasetDict(dict())

        self._set_multiset_verbose(verbose)

    @classmethod
    def _get_recipe_param(cls, param_dict, param_key: str, default_value=None):
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
                           replace: bool = False) -> None:
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
            self.stepC_prepped_datasets = le_datasets
        else:
            pass

    def __repr__(self):
        """ String representation is built.
        """
        strrep = f"Multiset: \n" + \
                 self._original_datasets_list_str() + \
                 f"\n" \
                 f"\t all attributes:%s" % '\n\t\t\t'.join(self._obj_attributes_list_str())

        return strrep

    def _obj_attributes_list_str(self) -> list:
        """ Get a list of each dataset attribute (with "empty" markers)
        """
        return sorted([f"{k}: empty" if (not self.__dict__[k]) else k
                       for k in self.__dict__.keys()
                       if not k.startswith('_')])

    def _original_datasets_list_str(self) -> str:
        """ Get a list of the identifying keys for each dataset
        """
        if self.stepA_original_datasets:
            return '\n\t'.join(self.stepA_original_datasets.keys())
        else:
            return ''

    def _set_multiset_verbose(self, verbose: Union[bool, str] = False) -> None:
        """This sets the verbosity level of the Multiset class only.

        Parameters
        ----------
        verbose
            either True, False, or a string for level such as "INFO, DEBUG, etc."

        """
        _multiset_logger.setLevel(self._validate_verbose(verbose))

    @staticmethod
    def _validate_verbose(verbose: Union[bool, str] = False) -> Union[int, str]:
        """

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
