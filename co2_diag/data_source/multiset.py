from co2_diag import set_verbose
from co2_diag.operations.anomalies import monthly_anomalies
from co2_diag.operations.datasetdict import DatasetDict
from co2_diag.operations.time import ensure_datetime64_array, ensure_dataset_datetime64
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Union
import pickle, logging

_multiset_logger = logging.getLogger("{0}.{1}".format(__name__, "multiset"))


class Multiset:
    """Useful class for working simultaneously with multiple, consistent xarray Datasets."""

    def __init__(self, verbose: Union[bool, str] = False):
        """
        This class is a template against which we can run recipes, with an order of operations:
            - Step A: datasets loaded, in their original form
            - Step B: datasets that have been preprocessed
            - Step C: datasets that have operations lazily queued or fully processed

        Parameters
        ----------
        verbose: Union[bool, str]
            either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        self.stepA_original_datasets: Union[DatasetDict, None] = None
        self.stepB_preprocessed_datasets: Union[DatasetDict, None] = None
        self.stepC_prepped_datasets: DatasetDict = DatasetDict(dict())

        set_verbose(_multiset_logger, verbose)

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

    def datasets_from_pickle(self,
                             filename: str = 'cmip_collection.latest_executed_datasets.pickle',
                             replace: bool = False) -> Union[bool, DatasetDict]:
        """Load a dataset dictionary from a saved pickle file.

        Parameters
        ----------
        filename: str
        replace: bool

        Returns
        -------
        If loaded successfully:
            True, if replace==True
            A DatasetDict if replace==False
        if loaded unsuccessfully:
            False

        Notes
        -----
        The pickle load protocol version used is detected automatically, so we do not have to specify it.
        """
        if not filename:
            return False

        with open(filename, 'rb') as f:
            _multiset_logger.debug('Loading dataset from file..')
            if replace:
                self.stepC_prepped_datasets = pickle.load(f)
                return True
            else:
                return pickle.load(f)

    def validate_time_options(self, starttime_option, endtime_option):
        """Check whether the specified start time is before the data's end time
            and the specified end time is after the data's start time.

        Parameters
        ----------
        starttime_option
        endtime_option

        Returns
        -------

        """
        for k, v in self.stepB_preprocessed_datasets.items():
            data_starttime = v['time'].min().values
            data_endtime = v['time'].max().values
            if starttime_option > data_endtime:
                raise ValueError("The specified start time (%s) is later than %s data's end time (%s)",
                                 starttime_option, k, data_endtime)
            elif endtime_option < data_starttime:
                raise ValueError("The specified end time (%s) is later than %s data's start time (%s)",
                                 endtime_option, k, data_starttime)

    @staticmethod
    def get_anomaly_dataframes(data: Union[xr.DataArray, xr.Dataset],
                               varname: str
                               ) -> (pd.DataFrame, pd.DataFrame):
        """

        Parameters
        ----------
        data
        varname

        Returns
        -------

        """
        if isinstance(data, xr.DataArray):
            data = ensure_datetime64_array(data)
        elif isinstance(data, xr.Dataset):
            data = ensure_dataset_datetime64(data)
        else:
            raise TypeError('Unexpected type <%s>. Was expecting either xarray.Dataset or xarray.DataArray',
                            type(data))

        # Calculate
        df_anomaly = monthly_anomalies(data, varname=varname)
        # Reformat data structures for plotting
        _df_anomaly_yearly = df_anomaly.pivot(index='moy', columns='year', values='monthly_anomaly_from_year')
        _df_anomaly_mean_cycle = df_anomaly.groupby('moy').mean().reset_index()

        return _df_anomaly_mean_cycle, _df_anomaly_yearly

    @staticmethod
    def categorical_cmap(nc: int,
                         nsc: int,
                         cmap: str = "tab10",
                         continuous: bool = False):
        """
        Parameters
        ----------
        nc: int
            number of categories
        nsc: int
            number of subcategories
        cmap: str
        continuous: bool

        Returns
        -------
            A colormap with nc*nsc different colors, where for each category there are nsc colors of same hue

        Notes
        -----
            from https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
        """
        if nc > plt.get_cmap(cmap).N:
            raise ValueError("Too many categories for colormap.")
        if continuous:
            ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
        else:
            ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
        cols = np.zeros((nc * nsc, 3))
        for i, c in enumerate(ccolors):
            chsv = mpl.colors.rgb_to_hsv(c[:3])
            arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
            arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
            arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
            rgb = mpl.colors.hsv_to_rgb(arhsv)
            cols[i * nsc:(i + 1) * nsc, :] = rgb
        cmap = mpl.colors.ListedColormap(cols)
        return cmap

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
        list_builder = []
        for k in self.__dict__.keys():
            if not k.startswith('_'):
                if isinstance(self.__dict__[k], pd.DataFrame) | isinstance(self.__dict__[k], pd.Series):
                    # Pandas object truth value can't be compared without .empty
                    if (not self.__dict__[k].empty):
                        list_builder.append(f"{k}: empty")
                    else:
                        list_builder.append(k)
                elif (not self.__dict__[k]):
                    list_builder.append(f"{k}: empty")
                else:
                    list_builder.append(f"{k}: {type(k)}")

        return sorted(list_builder)
        # return sorted([f"{k}: empty" if (not self.__dict__[k]) else k
        #                for k in self.__dict__.keys()
        #                if not k.startswith('_')])

    def _original_datasets_list_str(self) -> str:
        """ Get a list of the identifying keys for each dataset
        """
        if self.stepA_original_datasets:
            return '\n\t'.join(self.stepA_original_datasets.keys())
        else:
            return ''
