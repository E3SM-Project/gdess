import pickle
from typing import Union
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from co2_diag import validate_verbose
import co2_diag.data_source as co2ops
from co2_diag.data_source.datasetdict import DatasetDict

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
                           replace: bool = False) -> Union[bool, DatasetDict]:
        """Load a dataset dictionary from a saved pickle file.

        Parameters
        ----------
        filename
        replace

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
        _multiset_logger.info('Loading dataset from file..')
        if not filename:
            return False

        with open(filename, 'rb') as f:
            if replace:
                self.stepC_prepped_datasets = pickle.load(f)
                return True
            else:
                return pickle.load(f)

    @staticmethod
    def get_anomaly_dataframes(a_dataarray, varname: str):
        if not isinstance(a_dataarray['time'].values[0], np.datetime64):
            # Some time variables are numpy datetime64, some are CFtime.  Errors are raised if plotted together.
            a_dataarray = co2_diag.data_operation_utils.time.to_datetimeindex(a_dataarray)
        # Calculate
        df_anomaly = co2ops.obspack.anomalies.monthly_anomalies(a_dataarray, varname=varname)
        # Reformat data structures for plotting
        _df_anomaly_yearly = df_anomaly.pivot(index='moy', columns='year', values='monthly_anomaly_from_year')
        _df_anomaly_mean_cycle = df_anomaly.groupby('moy').mean().reset_index()

        return _df_anomaly_mean_cycle, _df_anomaly_yearly

    @staticmethod
    def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
        """
        Parameters
        ----------
        nc
            number of categories
        nsc
            number of subcategories
        cmap
        continuous

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

    def _set_multiset_verbose(self, verbose: Union[bool, str] = False) -> None:
        """This sets the verbosity level of the Multiset class only.

        Parameters
        ----------
        verbose
            either True, False, or a string for level such as "INFO, DEBUG, etc."

        """
        _multiset_logger.setLevel(validate_verbose(verbose))
