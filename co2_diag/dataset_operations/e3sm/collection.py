import numpy as np
import xarray as xr
import warnings
from typing import Union

import co2_diag.dataset_operations as co2ops
from co2_diag.dataset_operations.e3sm.calculation import getPINT, getPMID
from co2_diag.dataset_operations.multiset import Multiset, benchmark_recipe
from co2_diag.dataset_operations.datasetdict import DatasetDict
from co2_diag.dataset_operations.geographic import get_closest_mdl_cell_dict

from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig

# Packages for using NCAR's intake
import intake
import intake_esm

import matplotlib as mpl
import matplotlib.pyplot as plt

import logging
_loader_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))


class Collection(Multiset):
    def __init__(self, verbose=False):
        """

        Parameters
        ----------
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        self.set_verbose(verbose)

        super().__init__(verbose=verbose)

    @classmethod
    def _e3sm_recipe_base(cls,
                          verbose: Union[bool, str] = False,
                          load_from_file=None,
                          nc_file: str = None
                          ) -> ('Collection', bool):
        """Create an instance, and either preprocess or load already processed data.

        Parameters
        ----------
        verbose
        load_from_file
        nc_file

        Returns
        -------
        tuple
            Collection
            bool
                Whether datasets were loaded from file or not. (If not, there is probably more processing needed.)
        """
        # An empty instance is created.
        new_self = cls(verbose=verbose)

        # If a valid filename is provided, datasets are loaded into stepC attribute and this is True,
        # otherwise, this is False.
        loaded_from_file_bool = new_self.datasets_from_file(filename=load_from_file, replace=True)

        if not loaded_from_file_bool:
            # Data are formatted into the basic data structure common to various diagnostics.
            new_self.preprocess(filepath=nc_file)

        return new_self, loaded_from_file_bool

    @classmethod
    @benchmark_recipe
    def run_recipe_for_timeseries(cls,
                                  verbose: Union[bool, str] = False,
                                  load_from_file=None,
                                  param_kw: dict = None
                                  ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        load_from_file
            (str): path to pickled datastore
        param_kw
            An optional dictionary with zero or more of these parameter keys:
                test_data (str): path to NetCDF file of E3SM model output
                start_yr (str): '1960' is default
                end_yr (str): None is default
                lev (int): last value is default

        Returns
        -------
        Collection object for E3SM that was used to generate the diagnostic
        """
        _loader_logger.debug("Parsing diagnostic parameters ---")
        test_data = Multiset._get_recipe_param(param_kw, 'test_data', default_value=None)
        start_yr = Multiset._get_recipe_param(param_kw, 'start_yr', default_value="1960")
        end_yr = Multiset._get_recipe_param(param_kw, 'end_yr', default_value=None)

        new_self, loaded_from_file = cls._e3sm_recipe_base(verbose=verbose,
                                                           load_from_file=load_from_file,
                                                           nc_file=test_data)

        n_lev = len(new_self.stepB_preprocessed_datasets['main'].lev)  # get last level
        lev_index = Multiset._get_recipe_param(param_kw, 'lev_index', default_value=n_lev-1)

        results_dir = Multiset._get_recipe_param(param_kw, 'results_dir', default_value=None)

        # --- Apply diagnostic parameters and prep data for plotting ---
        if not loaded_from_file:
            _loader_logger.info('Applying selected bounds..')
            selection = {'time': slice(start_yr, end_yr)}
            new_self.stepC_prepped_datasets = new_self.stepB_preprocessed_datasets.queue_selection(**selection,
                                                                                                   inplace=False)
            iselection = {'lev': lev_index}
            new_self.stepC_prepped_datasets.queue_selection(**iselection, isel=True, inplace=True)
            # Spatial mean is calculated, leaving us with a time series.
            new_self.stepC_prepped_datasets.queue_mean(dim=('ncol'), inplace=True)
            # The lazily loaded selections and computations are here actually processed.
            new_self.stepC_prepped_datasets.execute_all(inplace=True)

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_timeseries()
        if results_dir:
            mysavefig(fig, results_dir, 'e3sm_timeseries', bbox_artists)

        return new_self

    def preprocess(self, filepath: str) -> None:
        """Set up the dataset that are common to every diagnostic

        Parameters
        ----------
        filepath
        """
        _loader_logger.info("Preprocessing ---")
        _loader_logger.debug('Opening the file..')
        self.stepA_original_datasets = DatasetDict({'main': xr.open_dataset(filepath)})
        self.stepB_preprocessed_datasets = self.stepA_original_datasets.copy()

        def preprocess_functions(dataset):
            dataset['PMID'] = getPMID(dataset['hyam'], dataset['hybm'], dataset['P0'], dataset['PS'])
            dataset = (dataset
                       .set_coords(['time', 'lat', 'lon', 'PMID'])
                       .sortby(['time'])
                       .pipe(co2ops.time.to_datetimeindex)
                       .pipe(co2ops.convert.co2_kgfrac_to_ppm, co2_var_name='CO2')
                       )
            return dataset
        _loader_logger.debug('setting coords, formatting time, converting to ppm..')
        self.stepB_preprocessed_datasets.apply_function_to_all(preprocess_functions, inplace=True)

        _loader_logger.info("Preprocessing done.")

    def plot_timeseries(self) -> (plt.Figure, plt.Axes, tuple):
        """Make timeseries plot of co2 concentrations from the E3SM output

        Requires self.stepC_prepped_datasets attribute with a time dimension.

        Returns
        -------
        matplotlib figure
        matplotlib axis
        tuple
            Extra matplotlib artists used for the bounding box (bbox) when saving a figure
        """
        my_cmap = self.categorical_cmap(nc=1, nsc=1, cmap="tab10")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

        plt.plot(self.stepC_prepped_datasets['main']['time'],
                 self.stepC_prepped_datasets['main']['CO2'],
                 label=f"E3SM simulation", color=my_cmap.colors[0], alpha=0.6)

        ax.set_ylabel('$CO_2$ [ppm]')
        aesthetic_grid_no_spines(ax)

        bbox_artists = ()

        return fig, ax, bbox_artists

    def set_verbose(self, verbose: Union[bool, str] = False) -> None:
        """
        Parameters
        ----------
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        _loader_logger.setLevel(self._validate_verbose(verbose))

    def __repr__(self) -> str:
        """ String representation is built.
        """
        strrep = f"-- E3SM Collection -- \n" \
                 f"Datasets:" \
                 f"\n\t" + \
                 self._original_datasets_list_str() + \
                 f"\n" \
                 f"All attributes:" \
                 f"\n\t" + \
                 '\n\t'.join(self._obj_attributes_list_str())

        return strrep
