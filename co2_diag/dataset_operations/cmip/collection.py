import numpy as np
import pandas as pd
import xarray as xr
import warnings
from typing import Union

from co2_diag import validate_verbose
import co2_diag.dataset_operations as co2ops
from co2_diag.dataset_operations.multiset import Multiset, benchmark_recipe
from co2_diag.dataset_operations.datasetdict import DatasetDict
from co2_diag.dataset_operations.geographic import get_closest_mdl_cell_dict

from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig

# Packages for using NCAR's intake
import intake
import intake_esm

import matplotlib.pyplot as plt

import logging
_loader_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))

default_cmip6_datastore_url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"


class Collection(Multiset):
    def __init__(self, datastore='cmip6', verbose=False):
        """

        Parameters
        ----------
        datastore
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        self.set_verbose(verbose)

        if datastore == 'cmip6':
            self.datastore_url = default_cmip6_datastore_url
        else:
            raise ValueError('Unexpected/unhandled datastore <%s>', datastore)

        self.latest_searched_model_catalog = None
        self.catalog_dataframe = None

        super().__init__(verbose=verbose)

    @classmethod
    def _cmip_recipe_base(cls,
                          datastore='cmip6',
                          verbose: Union[bool, str] = False,
                          load_from_file=None
                          ) -> ('Collection', bool):
        """Create an instance, and either preprocess or load already processed data.

        Parameters
        ----------
        datastore
        verbose
        load_from_file

        Returns
        -------
        tuple
            Collection
            bool
                Whether datasets were loaded from file or not. (If not, there is probably more processing needed.)
        """
        # An empty instance is created.
        new_self = cls(datastore=datastore, verbose=verbose)

        # If a valid filename is provided, datasets are loaded into stepC attribute and this is True,
        # otherwise, this is False.
        loaded_from_file_bool = new_self.datasets_from_file(filename=load_from_file, replace=True)

        if not loaded_from_file_bool:
            # Data are formatted into the basic data structure common to various diagnostics.
            new_self.preprocess(new_self.datastore_url)

        return new_self, loaded_from_file_bool

    @classmethod
    @benchmark_recipe
    def run_recipe_for_timeseries(cls,
                                  datastore='cmip6',
                                  verbose: Union[bool, str] = False,
                                  load_from_file=None,
                                  param_kw: dict = None
                                  ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        load_from_file
            (str): path to pickled datastore
        param_kw
            An optional dictionary with zero or more of these parameter keys:
                start_yr (str): '1960' is default
                end_yr (str): None is default
                plev (int): 100000 is default

        Returns
        -------
        Collection object for CMIP6 that was used to generate the diagnostic
        """
        new_self, loaded_from_file = cls._cmip_recipe_base(datastore=datastore, verbose=verbose,
                                                           load_from_file=load_from_file)

        _loader_logger.debug("Parsing diagnostic parameters ---")
        start_yr = Multiset._get_recipe_param(param_kw, 'start_yr', default_value="1960")
        end_yr = Multiset._get_recipe_param(param_kw, 'end_yr', default_value=None)
        plev = Multiset._get_recipe_param(param_kw, 'plev', default_value=100000)
        results_dir = Multiset._get_recipe_param(param_kw, 'results_dir', default_value=None)

        # --- Apply diagnostic parameters and prep data for plotting ---
        if not loaded_from_file:
            _loader_logger.info('Applying selected bounds..')
            selection = {'time': slice(start_yr, end_yr),
                         'plev': plev}
            new_self.stepC_prepped_datasets = new_self.stepB_preprocessed_datasets.queue_selection(**selection,
                                                                                                   inplace=False)
            # Spatial mean is calculated, leaving us with a time series.
            new_self.stepC_prepped_datasets.queue_mean(dim=('lon', 'lat'), inplace=True)
            # The lazily loaded selections and computations are here actually processed.
            new_self.stepC_prepped_datasets.execute_all(inplace=True)

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_timeseries()
        if results_dir:
            mysavefig(fig, results_dir, 'cmip_timeseries', bbox_artists)

        return new_self

    @classmethod
    @benchmark_recipe
    def run_recipe_for_vertical_profile(cls,
                                        datastore='cmip6',
                                        verbose: Union[bool, str] = False,
                                        load_from_file=None,
                                        param_kw: dict = None
                                        ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        load_from_file
            (str): path to pickled datastore
        param_kw
            An optional dictionary with zero or more of these parameter keys:
                start_yr (str): '1960' is default
                end_yr (str): None is default

        Returns
        -------
        Collection object for CMIP6 that was used to generate the diagnostic
        """
        new_self, loaded_from_file = cls._cmip_recipe_base(datastore=datastore, verbose=verbose,
                                                           load_from_file=load_from_file)

        _loader_logger.debug("Parsing diagnostic parameters ---")
        start_yr = Multiset._get_recipe_param(param_kw, 'start_yr', default_value="1960")
        end_yr = Multiset._get_recipe_param(param_kw, 'end_yr', default_value=None)
        results_dir = Multiset._get_recipe_param(param_kw, 'results_dir', default_value=None)

        # --- Apply diagnostic parameters and prep data for plotting ---
        if not loaded_from_file:
            _loader_logger.info('Applying selected bounds..')
            selection = {'time': slice(start_yr, end_yr)}
            new_self.stepC_prepped_datasets = new_self.stepB_preprocessed_datasets.queue_selection(**selection,
                                                                                                   inplace=False)
            # Mean is calculated, leaving us with a vertical profile.
            new_self.stepC_prepped_datasets.queue_mean(dim=('lon', 'lat', 'time'), inplace=True)
            # The lazily loaded selections and computations are here actually processed.
            new_self.stepC_prepped_datasets.execute_all(inplace=True)

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_vertical_profiles()
        if results_dir:
            mysavefig(fig, results_dir, 'cmip_vertical_plot', bbox_artists)

        return new_self

    @classmethod
    @benchmark_recipe
    def run_recipe_for_zonal_mean(cls,
                                  datastore='cmip6',
                                  verbose: Union[bool, str] = False,
                                  load_from_file=None,
                                  param_kw: dict = None
                                  ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        load_from_file
            (str): path to pickled datastore
        param_kw
            An optional dictionary with zero or more of these parameter keys:
                start_yr (str): '1960' is default
                end_yr (str): None is default

        Returns
        -------
        Collection object for CMIP6 that was used to generate the diagnostic
        """
        new_self, loaded_from_file = cls._cmip_recipe_base(datastore=datastore, verbose=verbose,
                                                           load_from_file=load_from_file)

        _loader_logger.debug("Parsing diagnostic parameters ---")
        start_yr = Multiset._get_recipe_param(param_kw, 'start_yr', default_value="1960")
        end_yr = Multiset._get_recipe_param(param_kw, 'end_yr', default_value=None)
        model_key = Multiset._get_recipe_param(param_kw, 'model_key', default_value=None)
        member_key = Multiset._get_recipe_param(param_kw, 'member_key', default_value=None)
        results_dir = Multiset._get_recipe_param(param_kw, 'results_dir', default_value=None)

        # --- Apply diagnostic parameters and prep data for plotting ---
        if not loaded_from_file:
            _loader_logger.info('Applying selected bounds..')
            selection = {'time': slice(start_yr, end_yr)}
            new_self.stepC_prepped_datasets = new_self.stepB_preprocessed_datasets.queue_selection(**selection,
                                                                                                   inplace=False)
            # The longitudinal mean will be calculated, leaving us with the average latitudinal/height gradients.
            new_self.stepC_prepped_datasets.queue_mean(dim=('lon', 'time'), inplace=True)
            # The lazily loaded selections and computations are here actually processed.
            new_self.stepC_prepped_datasets.execute_all(inplace=True)

        if ('member_key' not in locals()) or (not member_key):
            _loader_logger.debug("No 'member_key' supplied. Averaging over the available members: %s",
                                 new_self.stepC_prepped_datasets[model_key]['member_id'].values.tolist())
            member_key = new_self.stepC_prepped_datasets[model_key]['member_id'].values.tolist()

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_zonal_mean(model_key, member_key,
                                                         titlestr=f"{model_key} ({member_key})")
        if results_dir:
            mysavefig(fig, results_dir, 'cmip_zonal_mean_plot', bbox_artists)

        return new_self

    @classmethod
    @benchmark_recipe
    def run_recipe_for_annual_series(cls,
                                     datastore='cmip6',
                                     verbose: Union[bool, str] = False,
                                     load_from_file=None,
                                     param_kw: dict = None
                                     ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        load_from_file
            (str): path to pickled datastore
        param_kw
            An optional dictionary with zero or more of these parameter keys:
                start_yr (str): '1960' s default
                end_yr (str): None is default

        Returns
        -------
        Collection object for CMIP6 that was used to generate the diagnostic
        """
        new_self, loaded_from_file = cls._cmip_recipe_base(datastore=datastore, verbose=verbose,
                                                           load_from_file=load_from_file)

        _loader_logger.debug("Parsing diagnostic parameters ---")
        start_yr = Multiset._get_recipe_param(param_kw, 'start_yr', default_value="1960")
        end_yr = Multiset._get_recipe_param(param_kw, 'end_yr', default_value=None)
        plev = Multiset._get_recipe_param(param_kw, 'plev', default_value=100000)
        model_key = Multiset._get_recipe_param(param_kw, 'model_key', default_value=None)
        member_key = Multiset._get_recipe_param(param_kw, 'member_key', default_value=None)
        results_dir = Multiset._get_recipe_param(param_kw, 'results_dir', default_value=None)

        # --- Apply diagnostic parameters and prep data for plotting ---
        if not loaded_from_file:
            _loader_logger.info('Applying selected bounds..')
            selection = {'time': slice(start_yr, end_yr),
                         'plev': plev}
            new_self.stepC_prepped_datasets = new_self.stepB_preprocessed_datasets.queue_selection(**selection,
                                                                                                   inplace=False)
            # Spatial mean is calculated, leaving us with a time series.
            new_self.stepC_prepped_datasets.queue_mean(dim=('lon', 'lat'), inplace=True)
            # The lazily loaded selections and computations are here actually processed.
            new_self.stepC_prepped_datasets.execute_all(inplace=True)

        if ('member_key' not in locals()) or (not member_key):
            _loader_logger.debug("No 'member_key' supplied. Averaging over the available members: %s",
                                 new_self.stepC_prepped_datasets[model_key]['member_id'].values.tolist())
            member_key = new_self.stepC_prepped_datasets[model_key]['member_id'].values.tolist()

        # The mean is calculated across ensemble members if there are multiple.
        if isinstance(member_key, list) and (len(member_key) > 1):
            df_list_of_means = []
            df_list_of_yearly_cycles = []
            for mi, m in enumerate(member_key):
                darray = new_self.stepC_prepped_datasets[model_key].sel(member_id=m)
                df_anomaly_mean_cycle, df_anomaly_yearly = Multiset.get_anomaly_dataframes(darray, varname='co2')
                df_anomaly_yearly['member_id'] = m
                df_anomaly_mean_cycle['member_id'] = m
                df_list_of_means.append(df_anomaly_mean_cycle)
                df_list_of_yearly_cycles.append(df_anomaly_yearly)

            df_anomaly_mean_cycle = pd.concat(df_list_of_means).groupby(['moy', 'plev']).mean().reset_index()
            df_anomaly_yearly = pd.concat(df_list_of_yearly_cycles).groupby('moy').mean()
        else:
            darray = new_self.stepC_prepped_datasets[model_key].sel(member_id=member_key)
            df_anomaly_mean_cycle, df_anomaly_yearly = Multiset.get_anomaly_dataframes(darray, varname='co2')

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_annual_series(df_anomaly_yearly, df_anomaly_mean_cycle,
                                                            titlestr=f"{model_key} ({member_key})")
        if results_dir:
            mysavefig(fig, results_dir, 'cmip_annual_series', bbox_artists)

        return new_self

    def preprocess(self, url: str = default_cmip6_datastore_url) -> None:
        """Set up the dataset that are common to every diagnostic

        Parameters
        ----------
        url
        """
        _loader_logger.info("Preprocessing ---")

        _loader_logger.debug('Opening the ESM datastore catalog..')
        self.catalog_dataframe = intake.open_esm_datastore(url)

        _loader_logger.debug('Searching for model output subset..')
        esm_datastore = self._search(experiment_id='esm-hist', table_id=['Amon'], variable_id='co2')
        _loader_logger.debug(f"  {esm_datastore.df.shape[0]} model members identified")

        _loader_logger.debug('Loading model datasets into memory..')
        self._load_datasets_from_search()

        _loader_logger.info("Preprocessing done.")

    def _search(self, **query) -> intake_esm.core.esm_datastore:
        """Wrapper for intake's catalog search.

        Loads catalog into the attribute "latest_searched_model_catalog"

        query keyword arguments:
            experiment_id
            table_id
            variable_id
            institution_id
            member_id
            grid_label
        """
        _loader_logger.debug("query dictionary: %s", query)
        self.latest_searched_model_catalog = self.catalog_dataframe.search(**query)

        return self.latest_searched_model_catalog

    def _load_datasets_from_search(self) -> None:
        """Load datasets into memory."""
        # self.stepA_original_datasets = self.latest_searched_model_catalog.to_dataset_dict()
        self.stepA_original_datasets = DatasetDict(self.latest_searched_model_catalog.to_dataset_dict())

        self.stepB_preprocessed_datasets = self.stepA_original_datasets.copy()
        # Convert CO2 units to ppm
        _loader_logger.debug("Converting units to ppm..")
        self.stepB_preprocessed_datasets.apply_function_to_all(co2ops.convert.co2_molfrac_to_ppm,
                                                               co2_var_name='co2',
                                                               inplace=True)
        _loader_logger.debug("all converted.")
        # self.convert_all_to_ppm()
        _loader_logger.info("Model keys:")
        _loader_logger.info('\n'.join(self.stepA_original_datasets.keys()))

    # def convert_all_to_ppm(self):
    #     # Convert CO2 units to ppm
    #     _loader_logger.debug("Converting units to ppm..")
    #     self.apply_function_to_all_datasets(co2ops.convert.co2_molfrac_to_ppm, co2_var_name='co2')
    #     _loader_logger.debug("all converted.")

    @staticmethod
    def latlon_select(xr_ds: xr.Dataset,
                      lat: float,
                      lon: float,
                      ) -> xr.Dataset:
        """Select from dataset the column that is closest to specified lat/lon pair

        Parameters
        ----------
        xr_ds
        lat
        lon

        Returns
        -------

        """
        closest_point_dict = get_closest_mdl_cell_dict(xr_ds, lat=lat, lon=lon,
                                                       coords_as_dimensions=True)

        return xr_ds.stack(coord_pair=['lat', 'lon']).isel(coord_pair=closest_point_dict['index'])

    def plot_timeseries(self) -> (plt.Figure, plt.Axes, tuple):
        """Make timeseries plot of co2 concentrations from or more CMIP models

        Requires self.stepC_prepped_datasets attribute with a time dimension.

        Returns
        -------
        matplotlib figure
        matplotlib axis
        tuple
            Extra matplotlib artists used for the bounding box (bbox) when saving a figure
        """
        nmodels, member_counts = self._count_members()
        my_cmap = self.categorical_cmap(nc=len(member_counts), nsc=max(member_counts), cmap="tab10")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

        for ki, k in enumerate(self.stepC_prepped_datasets.keys()):
            for mi, m in enumerate(self.stepC_prepped_datasets[k]['member_id'].values.tolist()):
                color_count = ki * max(member_counts) + mi

                darray = self.stepC_prepped_datasets[k].sel(member_id=m)

                # Some time variables are numpy datetime64, some are CFtime.  Errors are raised if plotted together.
                if not isinstance(darray['time'].values[0], np.datetime64):
                    # Warnings are raised when converting CFtimes to datetimes, because subtle errors.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        darray = co2ops.time.to_datetimeindex(darray)

                ax.plot(darray['time'], darray.to_array().squeeze(), label=f"{k} ({m})",
                        color=my_cmap.colors[color_count], alpha=0.6)

        ax.set_ylabel('ppm')
        aesthetic_grid_no_spines(ax)

        leg = plt.legend(title='Models', frameon=False,
                         bbox_to_anchor=(1.05, 1), loc='upper left',
                         fontsize=12)
        bbox_artists = (leg,)

        return fig, ax, bbox_artists

    def plot_vertical_profiles(self) -> (plt.Figure, plt.Axes, tuple):
        """Make vertical profile plot of co2 concentrations.

        Returns
        -------
        matplotlib figure
        matplotlib axis
        tuple
            Extra matplotlib artists used for the bounding box (bbox) when saving a figure
        """
        nmodels, member_counts = self._count_members()
        my_cmap = self.categorical_cmap(nc=len(member_counts), nsc=max(member_counts), cmap="tab10")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

        for ki, k in enumerate(self.stepC_prepped_datasets.keys()):
            for mi, m in enumerate(self.stepC_prepped_datasets[k]['member_id'].values.tolist()):
                color_count = ki * max(member_counts) + mi

                darray = self.stepC_prepped_datasets[k].sel(member_id=m)
                ax.plot(darray['co2'].squeeze(), darray['plev'].squeeze(), label=f"{k} ({m})",
                        marker='o', linestyle='-',
                        color=my_cmap.colors[color_count], alpha=0.6)

        ax.invert_yaxis()
        ax.set_xlabel('ppm')
        ax.set_ylabel('pressure [Pa]')
        aesthetic_grid_no_spines(ax)

        leg = plt.legend(title='Models', frameon=False,
                         bbox_to_anchor=(1.05, 1), loc='upper left',
                         fontsize=12)
        bbox_artists = (leg,)

        return fig, ax, bbox_artists

    def plot_zonal_mean(self, model_key, member_key, titlestr) -> (plt.Figure, plt.Axes, tuple):
        """Make zonal mean plot of co2 concentrations.

        Returns
        -------
        matplotlib figure
        matplotlib axis
        tuple
            Extra matplotlib artists used for the bounding box (bbox) when saving a figure
        """
        # --- Extract a single model member  ---
        darray = self.stepC_prepped_datasets[model_key].sel(member_id=member_key)['co2']

        # --- Make Figure ---
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

        darray.plot.contourf(ax=ax, x='lat', y='plev', levels=40,
                             cbar_kwargs={'label': '$CO_2$ (ppm)', 'spacing': 'proportional'})
        #
        ax.invert_yaxis()
        ax.set_title(titlestr, fontsize=12)
        #
        ax.grid(linestyle='--', color='lightgray', alpha=0.3)
        for k in ax.spines.keys():
            ax.spines[k].set_alpha(0.5)

        bbox_artists = ()

        return fig, ax, bbox_artists

    @staticmethod
    def plot_annual_series(df_anomaly_yearly, df_anomaly_cycle, titlestr) -> (plt.Figure, plt.Axes, tuple):
        """Make timeseries plot with annual anomalies of co2 concentration.

        Returns
        -------
        matplotlib figure
        matplotlib axis
        tuple
            Extra matplotlib artists used for the bounding box (bbox) when saving a figure
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

        # ---- Plot Observations ----
        ax.plot(df_anomaly_yearly, label='annual cycle',
                color='#C0C0C0', linestyle='-', alpha=0.3, marker='.', zorder=-32)
        ax.plot(df_anomaly_cycle['moy'], df_anomaly_cycle['monthly_anomaly_from_year'],
                label='mean annual cycle', marker='o', zorder=10,
                color=(18 / 255, 140 / 255, 126 / 255))  # (255/255, 127/255, 14/255))
        #
        ax.set_ylim((-13, 7))
        #
        ax.set_ylabel('$CO_2$ (ppm)')
        ax.set_xlabel('month')
        ax.set_title(titlestr, fontsize=12)
        #
        #         ax.text(0.02, 0.92, f"{sc.upper()}, {station_dict[sc]['lat']:.1f}, {station_dict[sc]['lon']:.1f}",
        #                     horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        ax.text(0.02, 0.06, f"Global, surface level mean",
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        #
        # Define the legend
        handles, labels = ax.get_legend_handles_labels()
        display = (0, len(handles) - 1)
        leg = ax.legend([handle for i, handle in enumerate(handles) if i in display],
                        [label for i, label in enumerate(labels) if i in display],
                        loc='best', fontsize=12)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh._legmarker.set_alpha(1)
        #
        #         ax.grid(linestyle='--', color='lightgray')
        #         for k in ax.spines.keys():
        #             ax.spines[k].set_alpha(0.5)
        bbox_artists = (leg,)

        return fig, ax, bbox_artists

    def set_verbose(self, verbose: Union[bool, str] = False) -> None:
        """
        Parameters
        ----------
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        _loader_logger.setLevel(validate_verbose(verbose))

    def __repr__(self) -> str:
        """String representation is built."""
        nmodels, member_counts = self._count_members(verbose=False)
        strrep = f"-- CMIP Collection -- \n" \
                 f"Datasets:" \
                 f"\n\t" + \
                 self._original_datasets_list_str() + \
                 f"\n" + \
                 f"There are <{member_counts}> members for each of the {nmodels} models." \
                 f"\n" \
                 f"All attributes:" \
                 f"\n\t" + \
                 '\n\t'.join(self._obj_attributes_list_str())

        return strrep

    def _count_members(self, verbose=True):
        """Get the number of member_id values present for each model's dataset

        Parameters
        ----------
        verbose

        Returns
        -------
        Tuple
            The number of models (int) and the number of members for each model (list of int)
        """
        if self.stepA_original_datasets:
            ds_to_check = self.stepA_original_datasets
        elif self.stepB_preprocessed_datasets:
            ds_to_check = self.stepB_preprocessed_datasets
        elif self.stepC_prepped_datasets:
            ds_to_check = self.stepC_prepped_datasets
        else:
            return 0, 0

        member_counts = []
        for k in ds_to_check.keys():
            member_counts.append(len(ds_to_check[k]['member_id'].values))
        nmodels = len(member_counts)
        if verbose:
            _loader_logger.info(f"There are <%s> members for each of the %d models.", member_counts, nmodels)

        return nmodels, member_counts
