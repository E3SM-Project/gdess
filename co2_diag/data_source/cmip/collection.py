import argparse
import pandas as pd
import xarray as xr
import warnings
from typing import Union

from co2_diag import set_verbose
from co2_diag.data_source.multiset import Multiset
from co2_diag.data_source.datasetdict import DatasetDict
from co2_diag.operations.geographic import get_closest_mdl_cell_dict
from co2_diag.operations.time import ensure_dataset_datetime64, year_to_datetime64
from co2_diag.operations.convert import co2_molfrac_to_ppm
from co2_diag.recipes.utils import valid_year_string, options_to_args, benchmark_recipe, nullable_str

from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig

# Packages for using NCAR's intake
import intake
import intake_esm

import matplotlib.pyplot as plt

import logging
_loader_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))

default_cmip6_datastore_url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"


class Collection(Multiset):
    def __init__(self, datastore='cmip6', verbose: Union[bool, str] = False):
        """Instantiate a CMIP Collection object.

        Parameters
        ----------
        datastore: str
            a shortened name of an ESM catalog that we will query for model outputs.
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        # Set up the level of verbosity, i.e. how many log messages are displayed.
        set_verbose(_loader_logger, verbose)
        self._progressbar = True
        if _loader_logger.level > 10:  # 10 is debug, 20 is info, etc.
            self._progressbar = False

        self.latest_searched_model_catalog = None
        self.catalog_dataframe = None
        # TODO: add capability to handle additional datastores besides cmip6, such as CMIP5, etc.
        if datastore == 'cmip6':
            self.datastore_url = default_cmip6_datastore_url
        else:
            raise ValueError('Unexpected/unhandled datastore <%s>', datastore)

        super().__init__(verbose=verbose)

    @classmethod
    def _recipe_base(cls,
                     datastore='cmip6',
                     verbose: Union[bool, str] = False,
                     from_file: Union[bool, str] = None,
                     skip_selections: bool = False,
                     selection: dict = None,
                     mean_dims: tuple = None,
                     model_name: Union[str, list] = None
                     ) -> ('Collection', bool):
        """Create an instance, and either preprocess or load already processed data.

        Parameters
        ----------
        datastore: str
        verbose: Union[bool, str]
        from_file: Union[bool, str]
        selection: dict
        mean_dims: tuple

        Returns
        -------
        tuple: (Collection, bool)
            Collection
            bool
                Whether datasets were loaded from file or not. (If not, there is probably more processing needed.)
        """
        # An empty instance is created.
        new_self = cls(datastore=datastore, verbose=verbose)

        # If a valid filename is provided, datasets are loaded into stepC attribute and this is True,
        # otherwise, this is False.
        loaded_from_file_bool = new_self.datasets_from_file(filename=from_file, replace=True)

        if (not loaded_from_file_bool) & (not skip_selections):
            # Data are formatted into the basic data structure common to various diagnostics.
            new_self.preprocess(new_self.datastore_url, model_name)

            _loader_logger.debug(' applying selected bounds: %s', selection)
            new_self.stepC_prepped_datasets = new_self.stepB_preprocessed_datasets.queue_selection(**selection,
                                                                                                   inplace=False)
            # Spatial mean is calculated, leaving us with a time series.
            new_self.stepC_prepped_datasets.queue_mean(dim=mean_dims, inplace=True)
            # The lazily loaded selections and computations are here actually processed.
            new_self.stepC_prepped_datasets.execute_all(inplace=True)

        return new_self, loaded_from_file_bool

    @classmethod
    @benchmark_recipe
    def run_recipe_for_timeseries(cls,
                                  datastore='cmip6',
                                  verbose: Union[bool, str] = False,
                                  load_from_file: Union[bool, str] = None,
                                  options: dict = None
                                  ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore: str
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        load_from_file: Union[bool, str]
            path to pickled datastore
        options
            A dictionary with zero or more of these parameter keys:
                start_yr (str): '1960' is default
                end_yr (str): None is default
                plev (int): 100000 is default

        Returns
        -------
        Collection object for CMIP6 that was used to generate the diagnostic
        """
        set_verbose(_loader_logger, verbose)
        opts = _parse_options(options)

        # Apply diagnostic options and prep data for plotting
        selection = {'time': slice(opts.start_datetime, opts.end_datetime),
                     'plev': opts.plev}
        new_self, loaded_from_file = cls._recipe_base(datastore=datastore, verbose=verbose, from_file=load_from_file,
                                                      selection=selection, mean_dims=('lon', 'lat'),
                                                      model_name=opts.model_name)

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_timeseries()
        if opts.figure_savepath:
            mysavefig(fig, opts.figure_savepath, 'cmip_timeseries', bbox_extra_artists=bbox_artists)

        return new_self

    @classmethod
    @benchmark_recipe
    def run_recipe_for_vertical_profile(cls,
                                        datastore='cmip6',
                                        verbose: Union[bool, str] = False,
                                        load_from_file: Union[bool, str] = None,
                                        options: dict = None
                                        ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore: str
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        load_from_file: str
            path to pickled datastore
        options
            A dictionary with zero or more of these parameter keys:
                start_yr (str): '1960' is default
                end_yr (str): None is default

        Returns
        -------
        Collection object for CMIP6 that was used to generate the diagnostic
        """
        set_verbose(_loader_logger, verbose)
        opts = _parse_options(options)

        # Apply diagnostic options and prep data for plotting
        selection = {'time': slice(opts.start_datetime, opts.end_datetime)}
        new_self, loaded_from_file = cls._recipe_base(datastore=datastore, verbose=verbose, from_file=load_from_file,
                                                      selection=selection, mean_dims=('lon', 'lat', 'time'),
                                                      model_name=opts.model_name)

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_vertical_profiles()
        if opts.figure_savepath:
            mysavefig(fig, opts.figure_savepath, 'cmip_vertical_plot', bbox_extra_artists=bbox_artists)

        return new_self

    @classmethod
    @benchmark_recipe
    def run_recipe_for_zonal_mean(cls,
                                  datastore='cmip6',
                                  verbose: Union[bool, str] = False,
                                  load_from_file: Union[bool, str] = None,
                                  options: dict = None
                                  ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore: str
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        load_from_file: Union[bool, str]
            path to pickled datastore
        options
            A dictionary with zero or more of these parameter keys:
                start_yr (str): '1960' is default
                end_yr (str): None is default

        Returns
        -------
        Collection object for CMIP6 that was used to generate the diagnostic
        """
        set_verbose(_loader_logger, verbose)
        opts = _parse_options(options)

        # Apply diagnostic options and prep data for plotting
        selection = {'time': slice(opts.start_datetime, opts.end_datetime)}
        new_self, loaded_from_file = cls._recipe_base(datastore=datastore, verbose=verbose, from_file=load_from_file,
                                                      selection=selection, mean_dims=('lon', 'time'),
                                                      model_name=opts.model_name)

        if not opts.member_key:
            _loader_logger.debug("No 'member_key' supplied. Averaging over the available members: %s",
                                 new_self.stepC_prepped_datasets[opts.model_name]['member_id'].values.tolist())
            opts.member_key = new_self.stepC_prepped_datasets[opts.model_name]['member_id'].values.tolist()

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_zonal_mean(opts.model_name, opts.member_key,
                                                         titlestr=f"{opts.model_name} ({opts.member_key})")
        if opts.figure_savepath:
            mysavefig(fig, opts.figure_savepath, 'cmip_zonal_mean_plot', bbox_extra_artists=bbox_artists)

        return new_self

    @classmethod
    @benchmark_recipe
    def run_recipe_for_annual_series(cls,
                                     datastore='cmip6',
                                     verbose: Union[bool, str] = False,
                                     load_from_file: Union[bool, str] = None,
                                     options: dict = None
                                     ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore: str
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        load_from_file: Union[bool, str]
            path to pickled datastore
        options
            A dictionary with zero or more of these parameter keys:
                start_yr (str): '1960' s default
                end_yr (str): None is default

        Returns
        -------
        Collection object for CMIP6 that was used to generate the diagnostic
        """
        set_verbose(_loader_logger, verbose)
        opts = _parse_options(options)

        # Apply diagnostic options and prep data for plotting
        selection = {'time': slice(opts.start_datetime, opts.end_datetime),
                     'plev': opts.plev}
        new_self, loaded_from_file = cls._recipe_base(datastore=datastore, verbose=verbose, from_file=load_from_file,
                                                      selection=selection, mean_dims=('lon', 'lat'),
                                                      model_name=opts.model_name)

        if not opts.member_key:
            _loader_logger.debug("No 'member_key' supplied. Averaging over the available members: %s",
                                 new_self.stepC_prepped_datasets[opts.model_name]['member_id'].values.tolist())
            opts.member_key = new_self.stepC_prepped_datasets[opts.model_name]['member_id'].values.tolist()

        # The mean is calculated across ensemble members if there are multiple.
        if isinstance(opts.member_key, list) and (len(opts.member_key) > 1):
            df_list_of_means = []
            df_list_of_yearly_cycles = []
            for mi, m in enumerate(opts.member_key):
                _loader_logger.debug(' selecting model=%s, member=%s', opts.model_name, m)
                ds = new_self.stepC_prepped_datasets[opts.model_name].sel(member_id=m)

                df_anomaly_mean_cycle, df_anomaly_yearly = Multiset.get_anomaly_dataframes(ds, varname='co2')
                df_anomaly_yearly['member_id'] = m
                df_anomaly_mean_cycle['member_id'] = m
                df_list_of_means.append(df_anomaly_mean_cycle)
                df_list_of_yearly_cycles.append(df_anomaly_yearly)

            df_anomaly_mean_cycle = pd.concat(df_list_of_means).groupby(['moy', 'plev']).mean().reset_index()
            df_anomaly_yearly = pd.concat(df_list_of_yearly_cycles).groupby('moy').mean()
        else:
            darray = new_self.stepC_prepped_datasets[opts.model_name].sel(member_id=opts.member_key)
            df_anomaly_mean_cycle, df_anomaly_yearly = Multiset.get_anomaly_dataframes(darray, varname='co2')

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_annual_series(df_anomaly_yearly, df_anomaly_mean_cycle,
                                                            titlestr=f"{opts.model_name} ({opts.member_key})")
        if opts.figure_savepath:
            mysavefig(fig, opts.figure_savepath, 'cmip_annual_series', bbox_extra_artists=bbox_artists)

        return new_self

    def preprocess(self,
                   url: str = default_cmip6_datastore_url,
                   model_name: Union[str, list] = None
                   ) -> None:
        """Set up the datasets that are common to every diagnostic

        Parameters
        ----------
        url: str
        model_name: Union[str, list]
        """
        _loader_logger.debug("Preprocessing...")

        _loader_logger.debug(' Opening the ESM datastore catalog, using URL == %s', url)
        self.catalog_dataframe = intake.open_esm_datastore(url)

        """
        We use the intake package's search to load a catalog into the attribute "latest_searched_model_catalog"
        Acceptable search arguments include: 
            experiment_id
            table_id
            variable_id
            institution_id
            member_id
            grid_label
        """
        search_parameters = {'experiment_id': 'esm-hist',
                             'table_id': ['Amon'],
                             'variable_id': 'co2'}
        _loader_logger.debug(' Searching for model output subset, with parameters = %s', search_parameters)
        self.latest_searched_model_catalog = self.catalog_dataframe.search(**search_parameters)
        _loader_logger.debug(f"  {self.latest_searched_model_catalog.df.shape[0]} model members identified")

        self._load_datasets_from_search(model_name)

        _loader_logger.debug("Preprocessing is done.")

    def _load_datasets_from_search(self,
                                   model_name: list
                                   ) -> None:
        """Load datasets into memory."""
        _loader_logger.debug(' Loading into memory the following models: %s', model_name)
        self.stepA_original_datasets = DatasetDict(self.latest_searched_model_catalog.to_dataset_dict(progressbar=self._progressbar))
        # Extract all (or only the specified) datasets, and create a copy of each.
        if model_name:
            if isinstance(model_name, str):
                model_name = [model_name]
        else:
            model_name = self.stepA_original_datasets.keys()
        self.stepB_preprocessed_datasets = DatasetDict({k: self.stepA_original_datasets[k] for k in model_name})

        _loader_logger.debug("Converting units to ppm..")
        self.stepB_preprocessed_datasets.apply_function_to_all(co2_molfrac_to_ppm,
                                                               co2_var_name='co2',
                                                               inplace=True)
        self.stepB_preprocessed_datasets.apply_function_to_all(ensure_dataset_datetime64, inplace=True)
        _loader_logger.debug("all converted.")
        _loader_logger.debug("Keys for models that have been preprocessed:")
        _loader_logger.debug(' ' + '\n '.join(self.stepB_preprocessed_datasets.keys()))

    @staticmethod
    def latlon_select(xr_ds: xr.Dataset,
                      lat: float,
                      lon: float,
                      ) -> xr.Dataset:
        """Select from dataset the column that is closest to specified lat/lon pair

        Parameters
        ----------
        xr_ds: xr.Dataset
        lat: float
        lon: float

        Returns
        -------
        An xarray.Dataset
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

                data = self.stepC_prepped_datasets[k].sel(member_id=m)
                data = ensure_dataset_datetime64(data)

                y = data['co2'].squeeze()
                ax.plot(y['time'], y, label=f"{k} ({m})",
                        color=my_cmap.colors[color_count], alpha=0.6)

        ax.set_ylabel('ppm')
        aesthetic_grid_no_spines(ax)
        #
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

    def plot_zonal_mean(self, model_name, member_key, titlestr) -> (plt.Figure, plt.Axes, tuple):
        """Make zonal mean plot of co2 concentrations.

        Returns
        -------
        matplotlib figure
        matplotlib axis
        tuple
            Extra matplotlib artists used for the bounding box (bbox) when saving a figure
        """
        # --- Extract a single model member  ---
        darray = self.stepC_prepped_datasets[model_name].sel(member_id=member_key)['co2']

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

    def __repr__(self) -> str:
        """String representation is built."""
        nmodels, member_counts = self._count_members(suppress_log_message=True)
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

    def _count_members(self, suppress_log_message: bool = False):
        """Get the number of member_id values present for each model's dataset

        Parameters
        ----------
        suppress_log_message: bool

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
        if not suppress_log_message:
            _loader_logger.info(f"There are <%s> members for each of the %d models.", member_counts, nmodels)

        return nmodels, member_counts


# -- Define valid model choices --
model_choices = ['CMIP.CNRM-CERFACS.CNRM-ESM2-1.esm-hist.Amon.gr', 'CMIP.NCAR.CESM2.esm-hist.Amon.gn',
                 'CMIP.BCC.BCC-CSM2-MR.esm-hist.Amon.gn', 'CMIP.NOAA-GFDL.GFDL-ESM4.esm-hist.Amon.gr1']
def model_substring(s):
    """Function used to allow specification of model names by only supplying a partial string match

    Example
    -------
    >>> model_substring('BCC')
    returns 'CMIP.BCC.BCC-CSM2-MR.esm-hist.Amon.gn'
    """
    options = [c for c in model_choices if s in c]
    if len(options) == 1:
        return options[0]
    return s


def _parse_options(params: dict):
    _loader_logger.debug("Parsing diagnostic parameters...")

    param_argstr = options_to_args(params)
    _loader_logger.debug(' Parameter argument string == %s', param_argstr)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--ref_data', type=str)
    parser.add_argument('--figure_savepath', type=str, default=None)
    parser.add_argument('--start_yr', default="1960", type=valid_year_string)
    parser.add_argument('--end_yr', default="2015", type=valid_year_string)
    parser.add_argument('--plev', default=100000, type=int)
    parser.add_argument('--model_name', default=None,
                        type=model_substring, choices=model_choices)
    parser.add_argument('--member_key', default=None, type=nullable_str)
    args = parser.parse_args(param_argstr)

    # Convert times to numpy.datetime64
    args.start_datetime = year_to_datetime64(args.start_yr)
    args.end_datetime = year_to_datetime64(args.end_yr)

    _loader_logger.debug("Parsing is done. Parsed options: %s", args)
    return args
