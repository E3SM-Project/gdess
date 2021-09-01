""" This produces plots of seasonal cycles of atmospheric CO2
This function parses:
 - observational data from Globalview+ surface stations
 - model output from CMIP6
================================================================================
"""
from co2_diag import set_verbose, benchmark_recipe
from co2_diag.recipe_parsers import parse_recipe_options, add_meridional_args_to_parser
from co2_diag.recipes.recipe_utils import load_cmip_model_output, populate_station_list
from co2_diag.operations.Confrontation import Confrontation
from dask.diagnostics import ProgressBar
from typing import Union
import argparse, logging

_logger = logging.getLogger(__name__)


@benchmark_recipe
def meridional_gradient(options: Union[dict, argparse.Namespace],
                        verbose: Union[bool, str] = False,
                        ) -> tuple:
    """Execute a series of preprocessing steps and generate a diagnostic result.

    Relevant co2_diag collections are instantiated and processed.

    If one station is specified, then that will be compared against model data at the same location
    If more than one station is specified, then no model data will be compared against it.

    Parameters
    ----------
    options: Union[dict, argparse.Namespace]
        Recipe options specified as key:value pairs. It can contain the following keys:
            ref_data (str): Required. directory containing the NOAA Obspack NetCDF files
            model_name (str): 'CMIP.NOAA-GFDL.GFDL-ESM4.esm-hist.Amon.gr1' is default
            station_code (str): a three letter code to specify the desired surface observing station; 'mlo' is default
            cmip_load_method (str):
                either 'pangeo' (which uses a stored url),
                or 'local' (which uses the path defined in config file)
            start_yr (str): '1960' is default
            end_yr (str): '2015' is default
            latitude_bin_size (numeric): None is default
            figure_savepath (str): None is default
            difference (str): None is default
            region_name (str): calculate averages within the region
                (uses the name and coordinates defined in config file)
            globalmean (str):
                either 'station', which requires specifying the <station_code> parameter,
                or 'global', which will calculate a global mean
    verbose: Union[bool, str]
        can be either True, False, or a string for level such as "INFO, DEBUG, etc."

    Returns
    -------
    A tuple:
        A DataFrame containing the data that were plotted.
        A list of the data for each station
        A DataFrame containing the metadata for each station
        (and if a comparison with a model was made, then the datetimes and values are also part of the returned tuple)
    """
    set_verbose(_logger, verbose)
    if verbose:
        ProgressBar().register()
    _logger.debug("Parsing diagnostic parameters...")
    opts = parse_recipe_options(options, add_meridional_args_to_parser)

    stations_to_analyze = populate_station_list(opts.run_all_stations, opts.station_list)

    # --- Load CMIP model output ---
    compare_against_model, ds_mdl = load_cmip_model_output(opts.model_name, opts.cmip_load_method, verbose=verbose)

    # --- Observation data is processed for each station location. ---
    # data_dict, df_all_cycles, df_station_metadata = loop_through_stations_and_calculate_monthly_patterns(
    #     compare_against_model, ds_mdl, opts, stations_to_analyze, verbose)
    conf = Confrontation(compare_against_model, ds_mdl, opts, stations_to_analyze, verbose)
    cycles_of_each_station, df_all_cycles, df_station_metadata, xdata_gv, xdata_mdl, ydata_gv, ydata_mdl = conf.looper(how='seasonal')

    return df_all_cycles, cycles_of_each_station, df_station_metadata
