import time
import numpy as np
import xarray as xr
import warnings
from typing import Union

import co2_diag.dataset_operations as co2ops
from co2_diag.dataset_operations.multiset import Multiset
from co2_diag.dataset_operations.geographic import get_closest_mdl_cell_dict

# Packages for using NCAR's intake
import intake
import intake_esm

import matplotlib as mpl
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

        self.original_datasets = None
        self.datasets_prepped_for_execution = {}
        self.latest_executed_datasets = {}
        self.catalog_dataframe = None
        super(Multiset, self).__init__()

    def preprocess(self, url: str = default_cmip6_datastore_url):
        """Set up the dataset that are common to every diagnostic

        Parameters
        ----------
        url

        Returns
        -------

        """
        _loader_logger.debug("Preprocessing ---")
        _loader_logger.info('Opening the ESM datastore catalog..')
        self.catalog_dataframe = intake.open_esm_datastore(url)

        # --- Get model datasets ---
        _loader_logger.info('Searching for model output subset..')
        esm_datastore = self.search(experiment_id='esm-hist',
                                        table_id=['Amon'],
                                        variable_id='co2')
        _loader_logger.info(f"  {esm_datastore.df.shape[0]} model members identified")
        _loader_logger.info('Loading model datasets into memory..')
        self.load_datasets_from_search()

    @classmethod
    def run_recipe_for_timeseries(cls,
                                  datastore='cmip6',
                                  verbose: Union[bool, str] = False,
                                  load_from_file=None,
                                  param_kw: dict = None
                                  ):
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
        Collection object for CMIP6

        """
        start_time = time.time()

        # An instance of this CMIP6 Collection is created.
        new_self = cls(datastore=datastore, verbose=verbose)
        # Data are formatted into the basic data structure common to various diagnostics.
        new_self.preprocess(new_self.datastore_url)

        # --- Parse additional Parameters ---
        _loader_logger.debug("Parsing additional parameters ---")
        # Default values are given here.
        start_yr = "1960"
        end_yr = None
        plev = 100000
        if param_kw:
            if 'start_yr' in param_kw:
                start_yr = param_kw['start_yr']
            if 'end_yr' in param_kw:
                end_yr = param_kw['end_yr']
            if 'plev' in param_kw:
                plev = param_kw['end_yr']

        # --- Get the parsed dataset ---
        if load_from_file is not None:
            _loader_logger.info('Loading dataset from file..')
            new_self.datasets_from_file(filename=load_from_file, replace=True)
        else:
            # -----------------------------
            # --- Apply selected bounds ---
            # -----------------------------
            _loader_logger.info('Applying selected bounds..')
            # We will slice the data by time and pressure level.
            selection_dict = {'time': slice(start_yr, end_yr),
                              'plev': plev}
            new_self.apply_selection(**selection_dict)
            # The spatial mean will be calculated, leaving us with a time series.
            new_self.apply_mean(dim=('lon', 'lat'))
            # The lazily loaded selections and computations are here actually processed.
            new_self.execute_all()

        # Report the time this recipe took to execute.
        execution_time = (time.time() - start_time)
        _loader_logger.info('recipe execution time before plotting (seconds): ' + str(execution_time))

        # --- Plotting ---
        fig, ax = new_self.lineplots()

        return new_self

    def __repr__(self) -> str:
        obj_attributes = sorted([k for k in self.__dict__.keys()
                                 if not k.startswith('_')])

        nmodels, member_counts = self.count_members(verbose=False)

        # String representation is built.
        strrep = f"-- CMIP Collection -- \n" \
                 f"Datasets:" \
                 f"\n\t" + \
                 '\n\t'.join(self.original_datasets.keys()) + \
                 f"\n" + \
                 f"There are <{member_counts}> members for each of the {nmodels} models." \
                 f"\n" \
                 f"All attributes:" \
                 f"\n\t" + \
                 '\n\t'.join(obj_attributes)

        return strrep

    def set_verbose(self, verbose: Union[bool, str] = False):
        # verbose can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        _loader_logger.setLevel(self._validate_verbose(verbose))

    def search(self, **query) -> intake_esm.core.esm_datastore:
        """Wrapper for intake's catalog search.

        Loads catalog into the attribute "latest_searched_model_catalog"

        query keyword arguments:
            experiment_id
            table_id
            variable_id
            institution_id
            member_id
            grid_label

        Returns
        -------

        """
        _loader_logger.info("query dictionary: %s", query)
        self.latest_searched_model_catalog = self.catalog_dataframe.search(**query)

        return self.latest_searched_model_catalog

    def load_datasets_from_search(self):
        """Load datasets into memory
        Returns
        -------

        """
        self.original_datasets = self.latest_searched_model_catalog.to_dataset_dict()
        self.datasets_prepped_for_execution = self.original_datasets
        _loader_logger.info("Model keys:")
        _loader_logger.info('\n'.join(self.original_datasets.keys()))

        self.convert_all_to_ppm()

    def convert_all_to_ppm(self):
        # Convert CO2 units to ppm
        _loader_logger.debug("Converting units to ppm..")
        self.apply_function_to_all_datasets(co2ops.convert.co2_molfrac_to_ppm, co2_var_name='co2')
        _loader_logger.debug("all converted.")

    def count_members(self, verbose=True):
        # Get the number of member_id values present for each model's dataset.
        member_counts = []
        for k in self.original_datasets.keys():
            member_counts.append(len(self.original_datasets[k]['member_id'].values))
        nmodels = len(member_counts)
        if verbose:
            _loader_logger.info(f"There are <%s> members for each of the %d models.", member_counts, nmodels)

        return nmodels, member_counts

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

    @staticmethod
    def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
        """
        Params:
            nc: number of categories
            nsc: number of subcategories
            cmap:
            continuous:

        Returns:
            A colormap with nc*nsc different colors, where for each category there are nsc colors of same hue

        Notes:
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

    def lineplots(self):
        # plt.rcParams.update({'font.size': 12,
        #                      'lines.linewidth': 2,
        #                      })

        nmodels, member_counts = self.count_members()
        my_cmap = self.categorical_cmap(nc=len(member_counts), nsc=max(member_counts),
                                        cmap="tab10")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))

        for ki, k in enumerate(self.latest_executed_datasets.keys()):
            for mi, m in enumerate(self.latest_executed_datasets[k]['member_id'].values.tolist()):
                color_count = ki * max(member_counts) + mi

                darray = self.latest_executed_datasets[k].sel(member_id=m)

                # Some time variables are numpy datetime64, some are CFtime.  Errors are raised if plotted together.
                if isinstance(darray['time'].values[0], np.datetime64):
                    pass
                else:
                    # Warnings are raised when converting CFtimes to datetimes, because subtle errors.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        darray = co2ops.time.to_datetimeindex(darray)

                ax.plot(darray['time'], darray.to_array().squeeze(), label=f"{k} ({m})",
                        color=my_cmap.colors[color_count], alpha=0.6)

        ax.set_ylabel('ppm')
        ax.grid(True, linestyle='--', color='gray', alpha=1)
        for spine in ax.spines.values():
            spine.set_visible(False)

        leg = plt.legend(title='Models', frameon=False,
                         bbox_to_anchor=(1.05, 1), loc='upper left',
                         fontsize=12)

        plt.tight_layout()
        return fig, ax
