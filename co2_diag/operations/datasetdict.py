import xarray as xr
from dask.diagnostics import ProgressBar
import pickle, logging

_datasetdict_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))


class DatasetDict(dict):
    """A dict wrapper for working simultaneously with multiple, consistent xArray Datasets.

    Extends the 'dict' class to make it easy to apply selections and calculations
    to each and every Dataset in the dictionary.  Currently, the following procedures are supported:
        - selections
        - means
        - load
    """
    def __init__(self, *args, **kwargs):
        super(DatasetDict, self).__init__(*args, **kwargs)

    def queue_selection(self, **selection_dict):
        """Select from datasets.  Wrapper for xarray's .sel().

        Can also use xarray's .isel() with an additional argument.

        Example
        -------
        dsd = DatasetDict()

        One can pass slices or individual values:
            dsd.queue_selection(time=slice("1960", None), inplace=True)
            dsd.queue_selection(plev=100000, inplace=True)

        Selections can also be given as a dictionary by using the double splat operator:
            selection_dict = {'time': slice("1960", None),
                              'plev': 100000}
            new_dsd = dsd.queue_selection(**selection_dict, inplace=False)

        Parameters
        ----------
        selection_dict
            include <isel=True> to use index selection instead of keyword selection.

        Returns
        -------
            A DatasetDict with selections lazily queued, but not executed. Or None if inplace==True.
        """
        _datasetdict_logger.debug("Queueing selection operation. keyword args = %s", selection_dict)
        if selection_dict.pop('isel', False):
            returndict = self.apply_function_to_all(xr.Dataset.isel, **selection_dict)
        else:  # Use the standard selection method if 'isel' key exists & is false, or if key does not exist.
            returndict = self.apply_function_to_all(xr.Dataset.sel, **selection_dict)
        _datasetdict_logger.info("selection(s) queued, but not yet executed. Ready for .execute_all()")

        return returndict

    def queue_mean(self, dim, **kwargs):
        """Wrapper for calculating the mean for Xarray Datasets.

        Parameters
        ----------
        dim
        kwargs

        Returns
        -------
            A DatasetDict with selections lazily queued, but not executed. Or None if inplace==True.
        """
        _datasetdict_logger.debug("Queueing mean operation. keyword args = %s", kwargs)
        returndict = self.apply_function_to_all(xr.Dataset.mean, dim=dim, **kwargs)
        _datasetdict_logger.info("mean calculation queued for all, but not yet executed. Ready for .execute_all()")

        return returndict

    def apply_function_to_all(self, fnc, *args, **kwargs):
        """Helper for applying functions to multiple datasets.

        The specified function is queued lazily (unless executing=True) for execution on datasets
        of an origin dictionary, which will be copied to a destination dictionary.

        Hopefully with this, there shouldn't be a need to writing additional looping code.

        Parameters
        ----------
        fnc
        args
        kwargs:
            'inplace' (bool): whether the functions should be applied to this DatasetDict or
                whether a copy should be returned with the operations applied.

        Returns
        -------
            A DatasetDict if inplace==False, or None if inplace==True
        """
        _datasetdict_logger.debug("Processing datasets operation <%s>. keyword args = %s", fnc, kwargs)

        # The destination is either this instance or a copy (as determined by the 'inplace' keyword).
        #   Default is to create a copy.
        inplace = kwargs.pop('inplace', False)  # Key is removed once no longer needed.
        if inplace:
            destination_dict = self
        else:  # A copy is used if the 'inplace' key exists & is false, or if the key does not exist.
            destination_dict = self.copy()

        # The function is applied to each dataset.
        number_of_datasets = len(destination_dict)
        if number_of_datasets >= 1:
            for i, k in enumerate(destination_dict.keys()):
                _datasetdict_logger.debug("-- %d/%d - %s/.. ", i+1, number_of_datasets, k)
                destination_dict[k] = destination_dict[k].pipe(fnc, *args, **kwargs)
            _datasetdict_logger.debug("Operation processed on all datasets.")
        else:
            _datasetdict_logger.debug("Nothing done. No datasets are ready for execution.")

        if inplace:
            return None
        else:
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
            A DatasetDict if inplace==False, or None if inplace==True
        """
        if progressbar:
            ProgressBar().register()

        _datasetdict_logger.debug("Executing all queued functions.")
        returndict = self.apply_function_to_all(xr.Dataset.load, inplace=inplace)
        _datasetdict_logger.info("done.")

        return returndict

    def copy(self) -> 'DatasetDict':
        """Generate a new Datasetdict with each dataset copied
        Useful for preventing further operations from modifying the original.
        """
        new_datasetdict = DatasetDict()
        for k, v in self.items():
            new_datasetdict[k] = v.copy(deep=True)
        return new_datasetdict

    def to_pickle(self, filename: str = 'datasetdict.pickle',) -> None:
        """Pickle this DatasetDict using the highest protocol available.

        Parameters
        ----------
        filename
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def from_pickle(self,
                    filename: str = 'cmip_collection.latest_executed_datasets.pickle',
                    replace: bool = False) -> 'DatasetDict':
        """Load a DatasetDict from a saved pickle file.

        Parameters
        ----------
        filename
        replace
        """
        with open(filename, 'rb') as f:
            # The protocol version used is detected automatically, so we do not have to specify it.
            le_datasets = pickle.load(f)

        if replace:
            for k, v in le_datasets.items():
                self[k] = v
        else:
            return le_datasets
