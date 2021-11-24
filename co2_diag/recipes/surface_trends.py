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
            figure_savepath : str, default None
            difference : str, default None
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
    A dictionary containing the data that were plotted.
    """
    set_verbose(_logger, verbose)
    if verbose:
        ProgressBar().register()
    _logger.debug("Parsing diagnostic parameters...")
    opts = parse_recipe_options(options, add_surface_trends_args_to_parser)

    stations_to_analyze = populate_station_list(run_all_stations=False, station_list=opts.station_list)

    # --- Load CMIP model output ---
    compare_against_model, ds_mdl = load_cmip_model_output(opts.model_name, opts.cmip_load_method, verbose=verbose)

    conf = Confrontation(compare_against_model, ds_mdl, opts, stations_to_analyze, verbose)
    cycles_of_each_station, concatenated_dfs, df_station_metadata, \
        xdata_obs, xdata_mdl, ydata_obs, ydata_mdl, \
        rmse_y_true, rmse_y_pred = conf.looper(how='trend')

    # --- Create Graphic ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    diffs = {}
    if opts.difference:
        # Values at the same time
        for station in stations_to_analyze:
            merged = rmse_y_pred.loc[:, ['time', station]].merge(rmse_y_true.loc[:, ['time', station]],
                                                                 on='time', suffixes=("_pred", "_true"),)
            merged['diff'] = merged[station + '_pred'] - merged[station + '_true']
            # Plot
            ax.plot(merged['time'], merged['diff'],
                    label=f"model - obs [{station}]",
                    marker='.', linestyle='none')
            diffs[station] = merged['diff']
        #
        ax.set_ylim(limits_with_zero(ax.get_ylim()))
        #
        data_output = {'model': rmse_y_pred, 'obs': rmse_y_true, 'diff': diffs}

    else:
        data_output = {'model': ydata_mdl, 'obs': ydata_obs}
        #
        for station in stations_to_analyze:
            # Plot
            ax.plot(concatenated_dfs['ref']['time'], concatenated_dfs['ref'][station],
                    label=f"Obs [{station}]",
                    color='k')
            ax.plot(concatenated_dfs['mdl']['time'], concatenated_dfs['mdl'][station],
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
        savepath = append_before_extension(opts.figure_savepath, 'trend')
        mysavefig(fig=fig, plot_save_name=savepath,
                  bbox_inches='tight')
        _logger.info("Saved at <%s>" % savepath)
    if opts.data_savepath:
        savepath = append_before_extension(opts.data_savepath, 'trend')
        for k, v in data_output.items():
            fp = append_before_extension(savepath, str(k))
            v.to_csv(fp)
            _logger.info("Saved %s at <%s>" % (k, fp))

    return data_output


