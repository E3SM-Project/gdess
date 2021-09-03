""" This produces plots of seasonal cycles of atmospheric CO2
This function parses:
 - observational data from Globalview+ surface stations
 - model output from CMIP6
================================================================================
"""
from co2_diag import set_verbose, benchmark_recipe
from co2_diag.recipe_parsers import parse_recipe_options, add_seasonal_cycle_args_to_parser
from co2_diag.recipes.recipe_utils import populate_station_list
from co2_diag.graphics.comparison_plots import plot_comparison_against_model, plot_lines_for_all_station_cycles
from co2_diag.operations.Confrontation import Confrontation, load_cmip_model_output
from co2_diag.formatters import numstr, append_before_extension
from dask.diagnostics import ProgressBar
from typing import Union
import argparse, logging

_logger = logging.getLogger(__name__)


@benchmark_recipe
def seasonal_cycles(options: Union[dict, argparse.Namespace],
                    verbose: Union[bool, str] = False,
                    ) -> tuple:
    """Execute a series of preprocessing steps and generate a diagnostic result.

    Relevant co2_diag collections are instantiated and processed.

    If one station is specified, then that will be compared against model data at the same location
    If more than one station is specified, then no model data will be compared against it.

    Parameters
    ----------
    options : Union[dict, argparse.Namespace]
        Recipe options specified as key:value pairs. It can contain the following keys:
            ref_data : str
                (required) directory containing the NOAA Obspack NetCDF files
            model_name : str, default 'CMIP.NOAA-GFDL.GFDL-ESM4.esm-hist.Amon.gr1'
            cmip_load_method : str, default 'pangeo'
                either 'pangeo' (which uses a stored url),
                or 'local' (which uses the path defined in config file)
            start_yr : str, default '1960'
            end_yr : str, default '2015'
            latitude_bin_size : numeric, default None
            figure_savepath : str, default None
            difference : str, default None
            region_name : str
                calculate averages within the region (uses the name and coordinates defined in config file)
            globalmean : str
                either 'station', which requires specifying the <station_code> parameter,
                or 'global', which will calculate a global mean
            station_list : str, default 'mlo'
                a sequence of three letter codes (space-delimited) to specify
                the desired surface observing station
    verbose : Union[bool, str]
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
    opts = parse_recipe_options(options, add_seasonal_cycle_args_to_parser)

    stations_to_analyze = populate_station_list(opts.run_all_stations, opts.station_list)

    # --- Load CMIP model output ---
    compare_against_model, ds_mdl = load_cmip_model_output(opts.model_name, opts.cmip_load_method, verbose=verbose)

    conf = Confrontation(compare_against_model, ds_mdl, opts, stations_to_analyze, verbose)
    cycles_of_each_station, concatenated_dfs, df_station_metadata, \
        xdata_obs, xdata_mdl, ydata_obs, ydata_mdl, \
        rmse_y_true, rmse_y_pred = conf.looper(how='seasonal')

    # --- Plot the seasonal cycles at all station locations
    plot_lines_for_all_station_cycles(xdata_obs, ydata_obs.iloc[:, ::-1], figure_title="GV+",
                                      savepath=append_before_extension(opts.figure_savepath, 'obs_lineplot'))

    if ydata_mdl is not None:
        #   (ii) CMIP data
        plot_lines_for_all_station_cycles(xdata_obs, ydata_mdl.iloc[:, ::-1], figure_title="CMIP",
                                          savepath=append_before_extension(opts.figure_savepath, 'mdl_lineplot'))

        #   (iii) Model - obs difference
        ydiff = ydata_mdl - ydata_obs
        plot_lines_for_all_station_cycles(xdata_obs, ydiff.iloc[:, ::-1], figure_title="Difference",
                                          savepath=append_before_extension(opts.figure_savepath, 'diff_lineplot'))

        #   (iv) Model and obs difference
        plot_comparison_against_model(xdata_obs, ydata_obs, f'obs',
                                      xdata_obs, ydata_mdl, f'model',
                                      savepath=append_before_extension(opts.figure_savepath, 'overlapped'))

    return concatenated_dfs, cycles_of_each_station, df_station_metadata
