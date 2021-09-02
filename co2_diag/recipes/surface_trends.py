""" This produces a plot of multidecadal trends of atmospheric CO2
This function parses:
 - observational data from Globalview+ surface stations
 - model output from CMIP6
================================================================================
"""
from co2_diag import set_verbose, benchmark_recipe
from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig, limits_with_zero
from co2_diag.recipe_parsers import parse_recipe_options, add_surface_trends_args_to_parser
from co2_diag.recipes.recipe_utils import populate_station_list
from co2_diag.operations.Confrontation import Confrontation, load_cmip_model_output
from co2_diag.formatters import append_before_extension
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
from typing import Union
import logging

_logger = logging.getLogger(__name__)


@benchmark_recipe
def surface_trends(options: dict,
                   verbose: Union[bool, str] = False,
                   ):
    """Execute a series of preprocessing steps and generate a diagnostic result.

    Relevant co2_diag collections are instantiated and processed.

    Parameters
    ----------
    options: dict
        Recipe options specified as key:value pairs. It can contain the following keys:
            ref_data (str): Required. directory containing the NOAA Obspack NetCDF files
            model_name (str): 'CMIP.NOAA-GFDL.GFDL-ESM4.esm-hist.Amon.gr1' is default
            start_yr (str): '1960' is default
            end_yr (str): '2015' is default
            figure_savepath (str): None is default
            difference (str): None is default
            globalmean (str):
                either 'station', which requires specifying the <station_code> parameter,
                or 'global', which will calculate a global mean
            station_list
    verbose: Union[bool, str]
        can be either True, False, or a string for level such as "INFO, DEBUG, etc."

    Returns
    -------
    A dictionary containing the data that were plotted.
    """
    set_verbose(_logger, verbose)
    if verbose:
        ProgressBar().register()
    _logger.debug("Parsing diagnostic parameters...")
    opts = parse_recipe_options(options, add_surface_trends_args_to_parser)

    stations_to_analyze = populate_station_list(run_all_stations=False, station_list=opts.station_list)
    # TODO: make the below into a for loop to process numerous stations
    # station_code = stations_to_analyze[0]

    # --- Load CMIP model output ---
    compare_against_model, ds_mdl = load_cmip_model_output(opts.model_name, opts.cmip_load_method, verbose=verbose)

    # # --- Globalview+ data ---
    # _logger.info('*Processing Observations*')
    # obs_collection = obspack_surface_collection_module.Collection(verbose=verbose)
    # obs_collection.preprocess(datadir=opts.ref_data, station_name=station_code)
    # ds_obs = obs_collection.stepA_original_datasets[station_code]
    # _logger.info('%s', obs_collection.station_dict[station_code])
    #
    # # --- Globalview+ and CMIP are now handled together ---
    # da_obs, da_mdl = make_comparable(ds_obs, ds_mdl,
    #                                  time_limits=(np.datetime64(opts.start_yr), np.datetime64(opts.end_yr)),
    #                                  latlon=(ds_obs['latitude'].values[0], ds_obs['longitude'].values[0]),
    #                                  altitude=ds_obs['altitude'].values[0], altitude_method='lowest',
    #                                  global_mean=opts.globalmean, verbose=verbose)

    conf = Confrontation(compare_against_model, ds_mdl, opts, stations_to_analyze, verbose)
    cycles_of_each_station, df_all_cycles, df_station_metadata, xdata_obs, xdata_mdl, ydata_obs, ydata_mdl = conf.looper(how='trend')

    # --- Create Graphic ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if opts.difference:
        # Values at the same time
        # da_mdl_rs = ydata_mdl.resample(time="1MS").mean()
        # da_obs_rs = ydata_obs.resample(time="1MS").mean()['co2']
        #
        da_TestMinusRef = (ydata_mdl - ydata_obs)
        #
        data_output = {'model': ydata_mdl, 'obs': ydata_obs, 'diff': da_TestMinusRef}
        #
        # Plot
        ax.plot(xdata_obs, da_TestMinusRef,
                label='model - obs',
                marker='.', linestyle='none')
        #
        ax.set_ylim(limits_with_zero(ax.get_ylim()))

    else:
        # x_mdl = xdata_mdl[~np.isnan(ydata_mdl.values)]
        # y_mdl = ydata_mdl.values[~np.isnan(ydata_mdl.values)]
        #
        data_output = {'model': ydata_mdl, 'obs': ydata_obs}
        #
        # Plot
        ax.plot(xdata_obs, ydata_obs,
                # label=f'Obs [{station_code}]',
                color='k')
        ax.plot(xdata_mdl, ydata_mdl,
                label=f'Model [{opts.model_name}]',
                color='r', linestyle='-')

    ax.set_ylabel('$CO_2$ (ppm)')
    aesthetic_grid_no_spines(ax)
    #
    plt.legend()
    #
    plt.tight_layout()
    #
    if opts.figure_savepath:
        mysavefig(fig=fig, plot_save_name=append_before_extension(opts.figure_savepath, 'trend'),
                  bbox_inches='tight')

    return data_output


