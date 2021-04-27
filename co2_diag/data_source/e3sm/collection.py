import argparse
import xarray as xr
from typing import Union

from co2_diag import set_verbose
from co2_diag.data_source.e3sm.calculation import getPMID
from co2_diag.data_source.multiset import Multiset
from co2_diag.data_source.datasetdict import DatasetDict
from co2_diag.operations.time import to_datetimeindex, year_to_datetime64
from co2_diag.operations.convert import co2_kgfrac_to_ppm
from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig
from co2_diag.recipes.utils import benchmark_recipe, options_to_args, valid_year_string

import matplotlib.pyplot as plt

import logging
_loader_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))


class Collection(Multiset):
    def __init__(self, verbose: Union[bool, str] = False):
        """Instantiate an E3SM Collection object.

        Parameters
        ----------
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        set_verbose(_loader_logger, verbose)

        super().__init__(verbose=verbose)

    @classmethod
    def _recipe_base(cls,
                     verbose: Union[bool, str] = False,
                     from_file: Union[bool, str] = None,
                     nc_file: str = None
                     ) -> ('Collection', bool):
        """Create an instance, and either preprocess or load already processed data.

        Parameters
        ----------
        verbose: Union[bool, str]
        from_file: Union[bool, str]
        nc_file: str

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
        loaded_from_file_bool = new_self.datasets_from_file(filename=from_file, replace=True)

        if not loaded_from_file_bool:
            # Data are formatted into the basic data structure common to various diagnostics.
            new_self.preprocess(filepath=nc_file)

        return new_self, loaded_from_file_bool

    @classmethod
    @benchmark_recipe
    def run_recipe_for_timeseries(cls,
                                  verbose: Union[bool, str] = False,
                                  load_from_file=None,
                                  options: dict = None
                                  ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        load_from_file
            (str): path to pickled datastore
        options
            A dictionary with zero or more of these parameter keys:
                test_data (str): path to NetCDF file of E3SM model output
                start_yr (str): '1960' is default
                end_yr (str): None is default
                lev (int): last value is default

        Returns
        -------
        Collection object for E3SM that was used to generate the diagnostic
        """
        set_verbose(_loader_logger, verbose)
        opts = _parse_options(options)

        new_self, loaded_from_file = cls._recipe_base(verbose=verbose,
                                                      from_file=load_from_file,
                                                      nc_file=opts.test_data)
        n_lev = len(new_self.stepB_preprocessed_datasets['main'].lev)  # get last level
        if not opts.lev_index:
            opts.lev_index = n_lev-1

        # --- Apply diagnostic parameters and prep data for plotting ---
        if not loaded_from_file:
            _loader_logger.info('Applying selected bounds..')
            selection = {'time': slice(opts.start_datetime, opts.end_datetime)}
            new_self.stepC_prepped_datasets = new_self.stepB_preprocessed_datasets.queue_selection(**selection,
                                                                                                   inplace=False)
            iselection = {'lev': opts.lev_index}
            new_self.stepC_prepped_datasets.queue_selection(**iselection, isel=True, inplace=True)
            # Spatial mean is calculated, leaving us with a time series.
            new_self.stepC_prepped_datasets.queue_mean(dim=('ncol'), inplace=True)
            # The lazily loaded selections and computations are here actually processed.
            new_self.stepC_prepped_datasets.execute_all(inplace=True)

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_timeseries()
        if opts.figure_savepath:
            mysavefig(fig, opts.figure_savepath, 'e3sm_timeseries', bbox_extra_artists=bbox_artists)

        return new_self

    @staticmethod
    def _preprocess_functions(dataset: xr.Dataset):
        """Run a set of functions on a dataset

        - Set coordinates
        - Sort the time dimension
        - Ensure time is a datetimeindex type
        - Convert CO2 kg/kg to ppm

        Parameters
        ----------
        dataset: xr.Dataset

        Returns
        -------
        An xr.Dataset
        """
        dataset['PMID'] = getPMID(dataset['hyam'], dataset['hybm'], dataset['P0'], dataset['PS'])
        dataset = (dataset
                   .set_coords(['time', 'lat', 'lon', 'PMID'])
                   .sortby(['time'])
                   .pipe(to_datetimeindex)
                   .pipe(co2_kgfrac_to_ppm, co2_var_name='CO2')
                   )
        return dataset

    def preprocess(self, filepath: str) -> None:
        """Set up the dataset that are common to every diagnostic

        Parameters
        ----------
        filepath: str
        """
        _loader_logger.info("Preprocessing...")

        _loader_logger.debug(' Opening the file..')
        self.stepA_original_datasets = DatasetDict({'main': xr.open_dataset(filepath)})
        self.stepB_preprocessed_datasets = self.stepA_original_datasets.copy()

        _loader_logger.debug(' Setting coords, formatting time, converting to ppm..')
        self.stepB_preprocessed_datasets.apply_function_to_all(self._preprocess_functions, inplace=True)

        _loader_logger.info("Preprocessing is done.")

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
        #
        ax.set_ylabel('$CO_2$ [ppm]')
        aesthetic_grid_no_spines(ax)
        #
        bbox_artists = ()

        return fig, ax, bbox_artists

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


def _parse_options(params: dict):
    _loader_logger.debug("Parsing diagnostic parameters...")

    param_argstr = options_to_args(params)
    _loader_logger.debug(' Parameter argument string == %s', param_argstr)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--figure_savepath', type=str, default=None)
    parser.add_argument('--start_yr', default="1960", type=valid_year_string)
    parser.add_argument('--end_yr', default="2015", type=valid_year_string)
    parser.add_argument('--lev_index', default=None, type=int)
    args = parser.parse_args(param_argstr)

    # Convert times to numpy.datetime64
    args.start_datetime = year_to_datetime64(args.start_yr)
    args.end_datetime = year_to_datetime64(args.end_yr)

    _loader_logger.debug("Parsing is done.")
    return args
