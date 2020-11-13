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
        """
        self.original_datasets = None
        self.datasets_prepped_for_execution = {}
        self.latest_executed_datasets = {}
        if verbose:
            _multiset_logger.setLevel(logging.DEBUG)

    def __repr__(self):
        obj_attributes = sorted([k for k in self.__dict__.keys()
                                 if not k.startswith('_')])

        # String representation is built.
        strrep = f"Multiset: \n" + \
                 '\t\n'.join(self.original_datasets.keys()) + \
                 f"\n" \
                 f"\t all attributes:%s" % '\n\t\t\t'.join(obj_attributes)

        return strrep

    @staticmethod
    def set_verbose(switch):
        if switch == 'on':
            _multiset_logger.setLevel(logging.DEBUG)
        elif switch == 'off':
            _multiset_logger.setLevel(logging.WARN)
        else:
            raise ValueError("Unexpect/unhandled verbose option <%s>. Please use 'on' or 'off'", switch)

    def apply_function_to_all_datasets(self, fnc, *args, **kwargs):
        """

        Parameters
        ----------
        fnc
        args
        kwargs:
            'executing' (bool): specify whether we are executing calculations or lazily applying a new calculation

        Returns
        -------

        """

        destination_dict = self.datasets_prepped_for_execution
        if 'executing' in kwargs:
            if kwargs['executing']:
                destination_dict = self.latest_executed_datasets

        count = 0
        for k in self.datasets_prepped_for_execution.keys():
            count += 1
            _multiset_logger.debug("-- %d - {%s}.. ", count, k)
            destination_dict[k] = self.datasets_prepped_for_execution[k].pipe(fnc, *args, **kwargs)

        if count == 0:
            _multiset_logger.debug("Nothing done. No datasets are ready for execution.")
        else:
            _multiset_logger.debug("all processed.")

    def load_all(self, progressbar=True):
        if progressbar:
            ProgressBar().register()

        self.apply_function_to_all_datasets(xr.Dataset.load, executing=True)
        _multiset_logger.info("done.")

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

        Returns
        -------

        """
        _multiset_logger.debug("processing <%s>", selection_dict)
        self.apply_function_to_all_datasets(xr.Dataset.sel, **selection_dict)
        _multiset_logger.info("all selections applied, but not yet executed. Ready for .load()")

    def apply_mean(self, dim):
        self.apply_function_to_all_datasets(xr.Dataset.mean, dim=dim)
        _multiset_logger.info("mean applied to all, but not yet executed. Ready for .load()")
