from co2_diag import set_verbose, load_config_file
from co2_diag.data_source.multiset import Multiset
from co2_diag.operations.datasetdict import DatasetDict
from co2_diag.operations.time import ensure_dataset_datetime64
from co2_diag.operations.convert import co2_molfrac_to_ppm
from co2_diag.recipes.utils import benchmark_recipe, parse_recipe_options, add_shared_arguments_for_recipes
from co2_diag.formatters.args import nullable_str
from co2_diag.formatters import append_before_extension
from co2_diag.graphics.single_source_plots import plot_annual_series, plot_zonal_mean
from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig
import intake
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import Union, Sequence
import argparse, re, logging

_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))

default_cmip6_datastore_url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"

# -- Define valid model choices --
model_choices = ['CMIP.CNRM-CERFACS.CNRM-ESM2-1.esm-hist.Amon.gr', 'CMIP.NCAR.CESM2.esm-hist.Amon.gn',
                 'CMIP.BCC.BCC-CSM2-MR.esm-hist.Amon.gn', 'CMIP.NOAA-GFDL.GFDL-ESM4.esm-hist.Amon.gr1']
full_model_name_pattern = re.compile(
        r'(?P<activityid>[a-zA-Z\d\-]+)\.(?P<institutionid>[a-zA-Z\d\-]+)\.'
        r'(?P<sourceid>[a-zA-Z\d\-]+)\.(?P<experimentid>[a-zA-Z\d\-]+)\.'
        r'(?P<tableid>[a-zA-Z\d\-]+)\.(?P<gridlabel>[a-zA-Z\d\-]+)')

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
        set_verbose(_logger, verbose)
        self._progressbar = True
        if _logger.level > 10:  # 10 is debug, 20 is info, etc.
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
                     load_method: str = 'pangeo',
                     pickle_file: str = None,
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
        load_method: str
            either 'pangeo', 'local', or 'pickle'
        pickle_file: str
            (Optional) path to a saved pickle file is used if argument load_method=='pickle
        selection: dict
        mean_dims: tuple

        Returns
        -------
        tuple: (Collection, bool)
            Collection
            bool
                Whether datasets were loaded from a pickle file or not. (If not, there is probably more processing needed.)
        """
        # An empty instance is created.
        new_self = cls(datastore=datastore, verbose=verbose)
        _logger.debug(' skip_selections: %s', skip_selections)

        loaded_from_pickle_bool = False
        if load_method == 'pickle':
            # If a valid filename is provided, datasets are loaded into stepC attribute and this is True,
            # otherwise, this is False.
            loaded_from_pickle_bool = new_self.datasets_from_pickle(filename=pickle_file, replace=True)
            _logger.debug(' loaded from pickle? --> %s', loaded_from_pickle_bool)
        else:
            # Data are formatted into the basic data structure common to various diagnostics.
            new_self._load_data(method=load_method, url=new_self.datastore_url, model_name=model_name)
            new_self.preprocess()

            if not skip_selections:
                _logger.debug(' applying selected bounds: %s', selection)
                new_self.stepC_prepped_datasets = new_self.stepB_preprocessed_datasets.queue_selection(**selection,
                                                                                                       inplace=False)
                # Spatial mean is calculated, leaving us with a time series.
                new_self.stepC_prepped_datasets.queue_mean(dim=mean_dims, inplace=True)
                # The lazily loaded selections and computations are here actually processed.
                new_self.stepC_prepped_datasets.execute_all(inplace=True)

        return new_self, loaded_from_pickle_bool

    @classmethod
    @benchmark_recipe
    def run_recipe_for_timeseries(cls,
                                  datastore='cmip6',
                                  verbose: Union[bool, str] = False,
                                  pickle_file: str = None,
                                  options: dict = None
                                  ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore: str
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        pickle_file: str
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
        set_verbose(_logger, verbose)
        opts = parse_recipe_options(options, add_cmip_collection_args_to_parser)

        # Apply diagnostic options and prep data for plotting
        selection = {'time': slice(opts.start_datetime, opts.end_datetime),
                     'plev': opts.plev}
        new_self, _ = cls._recipe_base(datastore=datastore, verbose=verbose, pickle_file=pickle_file,
                                       selection=selection, mean_dims=('lon', 'lat'),
                                       model_name=opts.model_name, load_method=opts.cmip_load_method)

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_timeseries()
        if opts.figure_savepath:
            mysavefig(fig=fig, plot_save_name=append_before_extension(opts.figure_savepath, 'cmip_timeseries'),
                      bbox_extra_artists=bbox_artists)

        return new_self

    @classmethod
    @benchmark_recipe
    def run_recipe_for_vertical_profile(cls,
                                        datastore='cmip6',
                                        verbose: Union[bool, str] = False,
                                        pickle_file: str = None,
                                        options: dict = None
                                        ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore: str
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        pickle_file: str
            path to pickled datastore
        options
            A dictionary with zero or more of these parameter keys:
                start_yr (str): '1960' is default
                end_yr (str): None is default

        Returns
        -------
        Collection object for CMIP6 that was used to generate the diagnostic
        """
        set_verbose(_logger, verbose)
        opts = parse_recipe_options(options, add_cmip_collection_args_to_parser)

        # Apply diagnostic options and prep data for plotting
        selection = {'time': slice(opts.start_datetime, opts.end_datetime)}
        new_self, _ = cls._recipe_base(datastore=datastore, verbose=verbose, pickle_file=pickle_file,
                                       selection=selection, mean_dims=('lon', 'lat', 'time'),
                                       model_name=opts.model_name, load_method=opts.cmip_load_method)

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_vertical_profiles()
        if opts.figure_savepath:
            mysavefig(fig=fig, plot_save_name=append_before_extension(opts.figure_savepath, 'cmip_vertical_plot'),
                      bbox_extra_artists=bbox_artists)

        return new_self

    @classmethod
    @benchmark_recipe
    def run_recipe_for_zonal_mean(cls,
                                  datastore='cmip6',
                                  verbose: Union[bool, str] = False,
                                  pickle_file: str = None,
                                  options: dict = None
                                  ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore: str
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        pickle_file: str
            path to pickled datastore
        options
            A dictionary with zero or more of these parameter keys:
                start_yr (str): '1960' is default
                end_yr (str): None is default

        Returns
        -------
        Collection object for CMIP6 that was used to generate the diagnostic
        """
        set_verbose(_logger, verbose)
        opts = parse_recipe_options(options, add_cmip_collection_args_to_parser)

        # Apply diagnostic options and prep data for plotting
        selection = {'time': slice(opts.start_datetime, opts.end_datetime)}
        new_self, _ = cls._recipe_base(datastore=datastore, verbose=verbose, pickle_file=pickle_file,
                                       selection=selection, mean_dims=('lon', 'time'),
                                       model_name=opts.model_name, load_method=opts.cmip_load_method)

        if not opts.member_key:
            _logger.debug("No 'member_key' supplied. Averaging over the available members: %s",
                          new_self.stepC_prepped_datasets[opts.model_name]['member_id'].values.tolist())
            opts.member_key = new_self.stepC_prepped_datasets[opts.model_name]['member_id'].values.tolist()

        # --- Plotting ---
        #
        darray = new_self.stepC_prepped_datasets[opts.model_name].sel(member_id=opts.member_key)['co2']
        fig, ax, bbox_artists = plot_zonal_mean(darray, titlestr=f"{opts.model_name} ({opts.member_key})")
        if opts.figure_savepath:
            mysavefig(fig=fig, plot_save_name=append_before_extension(opts.figure_savepath, 'cmip_zonal_mean_plot'),
                      bbox_extra_artists=bbox_artists)

        return new_self

    @classmethod
    @benchmark_recipe
    def run_recipe_for_annual_series(cls,
                                     datastore='cmip6',
                                     verbose: Union[bool, str] = False,
                                     pickle_file: str = None,
                                     options: dict = None
                                     ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datastore: str
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        pickle_file: str
            path to pickled datastore
        options
            A dictionary with zero or more of these parameter keys:
                start_yr (str): '1960' s default
                end_yr (str): None is default

        Returns
        -------
        Collection object for CMIP6 that was used to generate the diagnostic
        """
        set_verbose(_logger, verbose)
        opts = parse_recipe_options(options, add_cmip_collection_args_to_parser)

        # Apply diagnostic options and prep data for plotting
        selection = {'time': slice(opts.start_datetime, opts.end_datetime),
                     'plev': opts.plev}
        new_self, _ = cls._recipe_base(datastore=datastore, verbose=verbose, pickle_file=pickle_file,
                                       selection=selection, mean_dims=('lon', 'lat'),
                                       model_name=opts.model_name, load_method=opts.cmip_load_method)
        ds = new_self.stepC_prepped_datasets[opts.model_name]

        if not opts.member_key:
            _logger.debug("No 'member_key' supplied. Averaging over the available members: %s",
                          new_self.stepC_prepped_datasets[opts.model_name]['member_id'].values.tolist())
            opts.member_key = new_self.stepC_prepped_datasets[opts.model_name]['member_id'].values.tolist()

        # The mean is calculated across ensemble members if there are multiple.
        if isinstance(opts.member_key, list) and (len(opts.member_key) > 1):
            df_list_of_means = []
            df_list_of_yearly_cycles = []
            for mi, m in enumerate(opts.member_key):
                _logger.debug(' selecting model=%s, member=%s', opts.model_name, m)
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
        fig, ax, bbox_artists = plot_annual_series(df_anomaly_yearly, df_anomaly_mean_cycle,
                                                   titlestr=f"{opts.model_name} ({opts.member_key})")
        ax.text(0.02, 0.06, f"Global, surface level mean",
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

        if opts.figure_savepath:
            mysavefig(fig=fig, plot_save_name=append_before_extension(opts.figure_savepath, 'cmip_annual_series'),
                      bbox_extra_artists=bbox_artists)

        return new_self

    def _load_data(self,
                   method: str = '',
                   url: str = default_cmip6_datastore_url,
                   model_name: Sequence[str] = None
                   ) -> None:
        """

        Parameters
        ----------
        method: str
            either 'pangeo' or 'local'
        url: str
            (Optional) only used if argument method=='pangeo'
        model_name: str
        """
        if method == 'pangeo':
            # --- Search for datasets in ESM data catalog ---
            _logger.debug(' Opening the ESM datastore catalog, using URL == %s', url)
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
                                 'variable_id': ['co2']}
            _logger.debug(' Searching for model output subset, with parameters = %s', search_parameters)
            self.latest_searched_model_catalog = self.catalog_dataframe.search(**search_parameters,
                                                                               require_all_on=["source_id"])
            _logger.debug(f"  {self.latest_searched_model_catalog.df.shape[0]} model members identified")

            # --- Load datasets into memory ---
            _logger.debug(' Loading into memory the following models: %s', model_name)
            self.stepA_original_datasets = DatasetDict(
                self.latest_searched_model_catalog.to_dataset_dict(progressbar=self._progressbar))
            # Extract all (or only the specified) datasets, and create a copy of each.
            if model_name:
                if isinstance(model_name, str):
                    model_name = [model_name]
            else:
                model_name = self.stepA_original_datasets.keys()
            self.stepA_original_datasets = DatasetDict({k: self.stepA_original_datasets[k] for k in model_name})

        if method == 'local':
            # A configuration object (for holding paths and settings) is read in to get the path to the data.
            config = load_config_file()
            cmip_data_path = config.get('CMIP', 'source')
            _logger.debug(f"Loading local CMIP output files from path <{cmip_data_path}>..")

            # NetCDF files are loaded. Each model has its own DatasetDict key.
            dd = DatasetDict()
            model_shortnames = ['MPI-ESM.esm-hist', 'BCC.esm-hist']
            for mdl_name in model_shortnames:
                mdl_name_dict = model_name_dict_from_valid_form(mdl_name)
                ds = xr.open_mfdataset(f"{cmip_data_path}/*{mdl_name_dict['sourceid']}*{mdl_name_dict['experimentid']}*.nc", decode_times=True)
                key = matched_model_and_experiment(ds.attrs['parent_source_id'] + '.' + ds.attrs['experiment_id'])
                dd[key] = ds

            self.stepA_original_datasets = dd

    def preprocess(self) -> None:
        """Set up the datasets that are common to every diagnostic
        """
        msg = "Preprocessing..."
        msg += "\nConverting units to ppm.."
        _logger.debug(msg)

        self.stepB_preprocessed_datasets = self.stepA_original_datasets.copy()
        self.stepB_preprocessed_datasets.apply_function_to_all(co2_molfrac_to_ppm, co2_var_name='co2', inplace=True)
        self.stepB_preprocessed_datasets.apply_function_to_all(ensure_dataset_datetime64, inplace=True)

        msg = "all converted."
        msg += "\nKeys for models that have been preprocessed:"
        msg += '\n ' + '\n '.join(self.stepB_preprocessed_datasets.keys())
        msg += "\nPreprocessing is done."
        _logger.debug(msg)

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
            _logger.info(f"There are <%s> members for each of the %d models.", member_counts, nmodels)

        return nmodels, member_counts


def add_cmip_collection_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """Add recipe arguments to a parser object

    Parameters
    ----------
    parser
    """
    add_shared_arguments_for_recipes(parser)
    parser.add_argument('--plev', default=100000, type=int)
    parser.add_argument('--model_name', default=None, type=matched_model_and_experiment, choices=model_choices)
    parser.add_argument('--member_key', default=None, type=nullable_str)
    parser.add_argument('--cmip_load_method', default='pangeo',
                        type=str, choices=['pangeo', 'local'])


def cmip_recipe_basics(func):
    """A decorator for starting a cmip recipe
    """
    def parse_and_run(*args, **kwargs):
        set_verbose(_logger, kwargs.get('verbose'))
        opts = parse_recipe_options(kwargs.get('options'), add_cmip_collection_args_to_parser)
        # Recipe is run.
        returnval = func(*args, **kwargs)

        return returnval
    return parse_and_run


def model_name_dict_from_valid_form(s: str) -> dict:
    """Transform model_name into a dictionary with the parts

    Parameters
    ----------
    s: str

    Raises
    ------
    ValueError, if the form of the input string does not match either form (1) or (2)
    """
    # The supplied string is expected to be either in a shortened form <source>.<experiment> or a full name.
    short_pattern = re.compile(
        r'(?P<sourceid>[a-zA-Z\d\-]+)\.(?P<experimentid>[a-zA-Z\d\-]+)')

    if match := short_pattern.search(s):
        return match.groupdict()
    elif match := full_model_name_pattern.search(s):
        return match.groupdict()
    else:
        raise ValueError("Expected at least a source_id with an experiment_id, in the form "
                         "<source_id>.<experiment_id>, e.g. 'BCC.esm-hist'.")


def matched_model_and_experiment(s: str) -> str:
    """Function used to allow specification of model names by only supplying a partial string match

    This function first checks whether the input is a string and of the form:
        (1) source_id.experiment_id
        or
        (2) activity_id.institution_id.source_id.experiment_id.table_id.grid_label
    A full name (i.e., in form (2)) will be returned, if the input matches one of the defined model choices.
    If the input does not match a defined model choice, then the input string will be returned unchanged.

    Example
    -------
    >>> matched_model_and_experiment('BCC.esm-hist')
    returns 'CMIP.BCC.BCC-CSM2-MR.esm-hist.Amon.gn'
    """
    # Transform the full names of the model choices into a dictionary of source and experiment ids.
    valid = [full_model_name_pattern.search(m).groupdict() for m in model_choices]
    valid_source_names = [v['sourceid'] for v in valid]

    # The supplied string is expected to be either in a shortened form <source>.<experiment> or a full name.
    if nullable_str(s):
        supplied = model_name_dict_from_valid_form(s)
    else:
        return s

    # match the substring to one of the full model names
    options = [(i, c) for i, c in enumerate(valid_source_names)
               if supplied['sourceid'] in c]
    if len(options) == 1:
        if valid[options[0][0]]['experimentid'] == supplied['experimentid']:
            return model_choices[options[0][0]]
    return s
