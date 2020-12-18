import copy
import pickle
from typing import Union
import xarray as xr
from dask.diagnostics import ProgressBar

import logging

_multiset_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))
class Multiset():
    """Useful class for working simultaneously with multiple, consistent xarray Datasets

    """
    def __init__(self, verbose=False):
        """

        Parameters
        ----------
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        self.original_datasets = None
        self.datasets_prepped_for_execution = {}
        self.latest_executed_datasets = {}

        self.set_verbose(verbose)

    def __repr__(self):
        obj_attributes = sorted([k for k in self.__dict__.keys()
                                 if not self.datasets_prepped_for_execution[k].startswith('_')])

        # String representation is built.
        strrep = f"Multiset: \n" + \
                 '\t\n'.join(self.original_datasets.keys()) + \
                 f"\n" \
                 f"\t all attributes:%s" % '\n\t\t\t'.join(obj_attributes)

        return strrep

    def set_verbose(self, verbose: Union[bool, str] = False):
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

    def datasets_to_file(self, filename: str = 'cmip_collection.latest_executed_datasets.pickle',):
        """

        Parameters
        ----------
        filename

        """
        with open(filename, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.latest_executed_datasets, f, pickle.HIGHEST_PROTOCOL)

    def datasets_from_file(self,
                           filename: str = 'cmip_collection.latest_executed_datasets.pickle',
                           replace: bool = False):
        """

        Parameters
        ----------
        filename
        replace

        """
        with open(filename, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            le_datasets = pickle.load(f)

        if replace:
            self.latest_executed_datasets = le_datasets
        else:
            pass

    def apply_function_to_all_datasets(self, fnc, *args, **kwargs):
        """Helper for applying functions to multiple datasets.

        Hopefully with this, there shouldn't be a need to writing additional looping code.

        Parameters
        ----------
        fnc
        args
        kwargs:
            'executing' (bool): specify whether we are executing calculations or lazily applying a new calculation
                This determines which attribute variable the results are saved into.
            '

        Returns
        -------

        """

        # Any keyword arguments are parsed.
        origin_dict = self.original_datasets
        if 'append' in kwargs:
            if kwargs['append']:
                origin_dict = self.datasets_prepped_for_execution
                # then remove key
                kwargs.pop('append', None)
        #
        destination_dict = self.datasets_prepped_for_execution
        if 'executing' in kwargs:
            if kwargs['executing']:
                destination_dict = self.latest_executed_datasets
                # then remove key
                kwargs.pop('executing', None)

        count = 0
        nd = len(origin_dict)
        for k in origin_dict.keys():
            count += 1
            _multiset_logger.debug("-- %d/%d - %s/.. ", count, nd, k)
            destination_dict[k] = origin_dict[k].pipe(fnc, *args, **kwargs)

        if count == 0:
            _multiset_logger.debug("Nothing done. No datasets are ready for execution.")
        else:
            _multiset_logger.debug("all processed.")

        return destination_dict

    def execute_all(self,
                    progressbar: bool = True,
                    inplace: bool = True):
        """Process any lazily loaded selections and computations

        Parameters
        ----------
        progressbar
        inplace

        Returns
        -------
            A copy of this Collection instance is returned if inplace=False,
                otherwise, None is returned.
        """
        if progressbar:
            ProgressBar().register()

        self.apply_function_to_all_datasets(xr.Dataset.load, executing=True)
        _multiset_logger.info("done.")

        if not inplace:
            return copy.deepcopy(self)

    def apply_selection(self, **selection_dict):
        """Select from dataset.  Wrapper for Xarray's .sel().

        Example
        -------
        One can pass slices or individual values:
            cmip_obj.apply_selection(time=slice("1960", None))
            cmip_obj.apply_selection(plev=100000)

        Selections can also be given as a dictionary by using the double splat operator:
            selection_dict = {'time': slice("1960", None),
                              'plev': 100000}
            cmip_obj.apply_selection(**selection_dict)

        Parameters
        ----------
        selection_dict
            include <isel=True> to use index selection instead of keyword selection.

        Returns
        -------
            a dictionary containing the datasets with selections lazily queued, but not executed.

        """
        _multiset_logger.debug("processing selection: <%s>", selection_dict)

        index_based_selection = False
        if 'isel' in selection_dict:
            index_based_selection = selection_dict['isel']
            # then remove isel from selections
            selection_dict.pop('isel', None)

        if index_based_selection:
            returndict = self.apply_function_to_all_datasets(xr.Dataset.isel, **selection_dict)
        else:
            returndict = self.apply_function_to_all_datasets(xr.Dataset.sel, **selection_dict)
        _multiset_logger.info("selection(s) applied, but not yet executed. Ready for .load()")

        return returndict

    def apply_mean(self, dim):
        self.apply_function_to_all_datasets(xr.Dataset.mean, dim=dim)
        _multiset_logger.info("mean applied to all, but not yet executed. Ready for .load()")
