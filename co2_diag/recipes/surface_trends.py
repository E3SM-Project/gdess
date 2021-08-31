""" This produces a plot of multidecadal trends of atmospheric CO2
This function parses:
 - observational data from Globalview+ surface stations
 - model output from CMIP6
================================================================================
"""
from co2_diag import set_verbose, benchmark_recipe
from co2_diag.operations.Confrontation import make_comparable
from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig, limits_with_zero
from co2_diag.recipe_parsers import parse_recipe_options, add_surface_trends_args_to_parser
import co2_diag.data_source.observations.gvplus_surface as obspack_surface_collection_module
import co2_diag.data_source.models.cmip.cmip_collection as cmip_collection_module
import numpy as np
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
            station_code (str): a three letter code to specify the desired surface observing station; 'mlo' is default
            start_yr (str): '1960' is default
            end_yr (str): '2015' is default
            figure_savepath (str): None is default
            difference (str): None is default
            globalmean (str):
                either 'station', which requires specifying the <station_code> parameter,
                or 'global', which will calculate a global mean
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

    # --- Globalview+ data ---
    _logger.info('*Processing Observations*')
    obs_collection = obspack_surface_collection_module.Collection(verbose=verbose)
    obs_collection.preprocess(datadir=opts.ref_data, station_name=opts.station_code)
    ds_obs = obs_collection.stepA_original_datasets[opts.station_code]
    _logger.info('%s', obs_collection.station_dict[opts.station_code])

    # --- CMIP data ---
    _logger.info('*Processing CMIP model output*')
    cmip_collection = cmip_collection_module.Collection(verbose=verbose)
    new_self, _ = cmip_collection._recipe_base(datastore='cmip6', verbose=verbose,
                                                              pickle_file=None, skip_selections=True)
    ds_mdl = new_self.stepB_preprocessed_datasets[opts.model_name]

    # --- Globalview+ and CMIP are now handled together ---
    da_obs, da_mdl = make_comparable(ds_obs, ds_mdl,
                                     time_limits=(np.datetime64(opts.start_yr), np.datetime64(opts.end_yr)),
                                     latlon=(ds_obs['latitude'].values[0], ds_obs['longitude'].values[0]),
                                     altitude=ds_obs['altitude'].values[0], altitude_method='lowest',
                                     global_mean=opts.globalmean, verbose=verbose)

    # --- Create Graphic ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if opts.difference:
        # Values at the same time
        da_mdl_rs = da_mdl.resample(time="1MS").mean()
        da_obs_rs = ds_obs.resample(time="1MS").mean()['co2']
        #
        da_TestMinusRef = (da_mdl_rs - da_obs_rs).dropna(dim='time')
        #
        data_output = {'model': da_mdl_rs, 'obs': da_obs_rs, 'diff': da_TestMinusRef}
        #
        # Plot
        ax.plot(da_TestMinusRef['time'], da_TestMinusRef,
                label='model - obs',
                marker='.', linestyle='none')
        #
        ax.set_ylim(limits_with_zero(ax.get_ylim()))

    else:
        x_mdl = da_mdl['time'][~np.isnan(da_mdl.values)]
        y_mdl = da_mdl.values[~np.isnan(da_mdl.values)]
        #
        data_output = {'model': da_mdl, 'obs': ds_obs}
        #
        # Plot
        ax.plot(ds_obs['time'], ds_obs['co2'],
                label=f'Obs [{opts.station_code}]',
                color='k')
        ax.plot(x_mdl, y_mdl,
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
        mysavefig(fig=fig, plot_save_name=opts.figure_savepath)

    return data_output


