""" This produces a plot of multidecadal trends of atmospheric CO2
This function parses:
 - observational data from Globalview+ surface stations
 - model output from CMIP6
================================================================================
"""
import argparse
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

from co2_diag import set_verbose
import co2_diag.data_source.obspack.surface_stations.collection as obspack_surface_collection_module
import co2_diag.data_source.cmip.collection as cmip_collection_module
from co2_diag.operations.geographic import get_closest_mdl_cell_dict
from co2_diag.operations.time import ensure_dataset_datetime64, year_to_datetime64
from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig, limits_with_zero
from co2_diag.recipes.utils import valid_year_string, options_to_args

import logging
_logger = logging.getLogger(__name__)


def surface_trends(verbose: Union[bool, str] = False,
                   options: dict = None,
                   ):
    """Execute a series of preprocessing steps and generate a diagnostic result.

    Relevant co2_diag collections are instantiated and processed.

    Parameters
    ----------
    verbose: Union[bool, str]
        can be either True, False, or a string for level such as "INFO, DEBUG, etc."
    options: dict
        Recipe options specified as key:value pairs. It can contain the following keys:
            ref_data (str): Required. directory containing the NOAA Obspack NetCDF files
            model_name (str): 'mlo' is default
            start_yr (str): '1960' is default
            end_yr (str): '2015' is default
            figure_savepath (str): None is default
            difference (str): None is default
            globalmean (str):
                either 'station', which requires specifying the <station_code> parameter,
                or 'global', which will calculate a global mean
            station_code (str): a three letter code to specify the desired surface observing station

    Returns
    -------
    A dictionary containing the data that were plotted.
    """
    set_verbose(_logger, verbose)
    if verbose:
        ProgressBar().register()
    opts = _parse_options(options)

    # --- Surface observations ---
    _logger.info('*Processing Observations*')
    obs_collection = obspack_surface_collection_module.Collection(verbose=verbose)
    obs_collection.preprocess(datadir=opts.ref_data, station_name=opts.station_code)
    ds_obs = obs_collection.stepA_original_datasets[opts.station_code]
    _logger.info('%s', obs_collection.station_dict[opts.station_code])

    # --- CMIP model output ---
    _logger.info('*Processing CMIP model output*')
    cmip_collection = cmip_collection_module.Collection(verbose=verbose)
    new_self, loaded_from_file = cmip_collection._recipe_base(datastore='cmip6', verbose=verbose,
                                                              from_file=None, skip_selections=True)
    ds_mdl = new_self.stepB_preprocessed_datasets[opts.model_name]

    # --- Obspack and CMIP are now handled Together ---
    _logger.info('Selected bounds for both:')
    # Time boundaries
    ds_obs = ensure_dataset_datetime64(ds_obs)
    ds_obs = ds_obs.where(ds_obs.time >= opts.start_datetime, drop=True)
    ds_obs = ds_obs.where(ds_obs.time <= opts.end_datetime, drop=True)
    #
    ds_mdl = ensure_dataset_datetime64(ds_mdl)
    ds_mdl = ds_mdl.where(ds_mdl.time >= opts.start_datetime, drop=True)
    ds_mdl = ds_mdl.where(ds_mdl.time <= opts.end_datetime, drop=True)
    _logger.info('  -- time>=%s  &  time<=%s', opts.start_datetime, opts.end_datetime)

    _logger.info('Selected bounds for CMIP:')
    _logger.info('  -- model=%s', opts.model_name)
    # Only the first ensemble member is selected, if there are more than one
    # (TODO: enable the selection of a specific ensemble member)
    if 'member_id' in ds_mdl['co2'].coords:
        ds_mdl = (ds_mdl
                  .isel(member_id=0)
                  .copy())
        _logger.info('  -- member_id=0')
    else:
        ds_mdl = ds_mdl.copy()

    # Surface values are selected.
    ds_mdl = ds_mdl['co2'].isel(plev=0)
    _logger.info('  -- plev=0')

    # A specific lat/lon is selected, or a global mean is calculated.
    if opts.globalmean:
        da_mdl = ds_mdl.mean(dim=('lat', 'lon'))
        _logger.info('  -- mean over lat and lon dimensions')
    else:
        mdl_cell = get_closest_mdl_cell_dict(new_self.stepB_preprocessed_datasets[opts.model_name],
                                             lat=obs_collection.station_dict[opts.station_code]['lat'],
                                             lon=obs_collection.station_dict[opts.station_code]['lon'],
                                             coords_as_dimensions=True)
        da_mdl = (ds_mdl
                  .where(ds_mdl.lat == mdl_cell['lat'], drop=True)
                  .where(ds_mdl.lon == mdl_cell['lon'], drop=True)
                  )
        _logger.info('  -- lat=%s', mdl_cell['lat'])
        _logger.info('  -- lon=%s', mdl_cell['lon'])

    # Lazy computations are executed.
    _logger.info('Applying selected bounds...')
    da_mdl = da_mdl.squeeze().compute()
    _logger.info('done.')

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


def _parse_options(params: dict):
    _logger.debug("Parsing diagnostic parameters...")

    param_argstr = options_to_args(params)
    _logger.debug('Parameter argument string == %s', param_argstr)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--ref_data', type=str)
    parser.add_argument('--model_name', default='CMIP.NOAA-GFDL.GFDL-ESM4.esm-hist.Amon.gr1',
                        type=cmip_collection_module.model_substring, choices=cmip_collection_module.model_choices)
    parser.add_argument('--start_yr', default="1960", type=valid_year_string)
    parser.add_argument('--end_yr', default="2015", type=valid_year_string)
    parser.add_argument('--figure_savepath', type=str, default=None)
    parser.add_argument('--station_code', default='mlo',
                        type=str, choices=obspack_surface_collection_module.station_dict.keys())
    parser.add_argument('--difference', action='store_true')
    parser.add_argument('--globalmean', action='store_true')
    args = parser.parse_args(param_argstr)

    # Convert times to numpy.datetime64
    args.start_datetime = year_to_datetime64(args.start_yr)
    args.end_datetime = year_to_datetime64(args.end_yr)

    _logger.debug("Parsing is done.")
    return args
