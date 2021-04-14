""" This produces a plot of multidecadal trends of atmospheric CO2
This function parses:
 - observational data from Globalview+ surface stations
 - model output from CMIP6
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

from co2_diag import validate_verbose
from co2_diag.recipes.utils import get_recipe_param
import co2_diag.data_source.obspack.surface_stations.collection as obspack_surface_collection_module
import co2_diag.data_source.cmip.collection as cmip_collection_module
from co2_diag.operations.geographic import get_closest_mdl_cell_dict
from co2_diag.operations.time import ensure_dataset_datetime64
from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig, limits_with_zero

import logging
_logger = logging.getLogger(__name__)


def surface_trends(verbose=False,
                   param_kw: dict = None,
                   ):
    """Execute a series of preprocessing steps and generate a diagnostic result.

    Relevant co2_diag collections are instantiated and processed.

    Parameters
    ----------
    verbose
        can be either True, False, or a string for level such as "INFO, DEBUG, etc."
    param_kw
        A dictionary with the following parameter keys:
            ref_data (str): directory containing the NOAA Obspack NetCDF files
            model_name (str): 'brw' is default
            start_yr (str): '1960' is default
            end_yr (str): '2015' is default
            savepath_figure (str): None is default
            station_or_global (str):
                either 'station', which requires specifying the <station_code> parameter,
                or 'global', which will calculate a global mean
            station_code (str): a three letter code to specify the desired surface observing station

    Returns
    -------
    A DataArray containing the values plotted.
    """
    _logger.setLevel(validate_verbose(verbose))
    # --- Recipe parameters are parsed. ---
    _logger.debug("Parsing parameter options...")
    ref_data = get_recipe_param(param_kw, 'ref_data', default_value=None)
    model_name = get_recipe_param(param_kw, 'model_name', default_value='GFDL')
    start_datetime = np.datetime64(get_recipe_param(param_kw, 'start_yr', default_value="1960"), 'D')
    end_datetime = np.datetime64(get_recipe_param(param_kw, 'end_yr', default_value="2015"), 'D')
    savepath_figure = get_recipe_param(param_kw, 'savepath_figure', default_value=None)
    absolute_or_difference = get_recipe_param(param_kw, 'absolute_or_difference', default_value='absolute')
    station_or_globalmean = get_recipe_param(param_kw, 'station_or_globalmean', default_value='station')
    # For a single station, we also check that it is accounted for in the class attribute dict.
    station_code = get_recipe_param(param_kw, 'station_code', default_value='mlo')
    if station_code not in obspack_surface_collection_module.station_dict:
        raise ValueError('Unexpected station name <%s>', param_kw['stationshortname'])
    _logger.debug("DONE.")

    # --- Surface observations ---
    _logger.info('*Processing Observations*')
    obs_collection = obspack_surface_collection_module.Collection(verbose=verbose)
    obs_collection.preprocess(datadir=ref_data, station_name=station_code)
    ds_obs = obs_collection.stepA_original_datasets[station_code]
    _logger.info('%s', obs_collection.station_dict[station_code])

    # --- CMIP model output ---
    _logger.info('*Processing CMIP model output*')
    cmip_collection = cmip_collection_module.Collection(verbose=verbose)
    new_self, loaded_from_file = cmip_collection._cmip_recipe_base(datastore='cmip6', verbose=verbose,
                                                                   load_from_file=None)
    ds_mdl = new_self.stepB_preprocessed_datasets[model_name]

    # --- Obspack and CMIP are now handled Together ---
    if verbose:
        ProgressBar().register()
    _logger.info('Selected bounds for both:')

    # Time boundaries
    ds_obs = ensure_dataset_datetime64(ds_obs)
    ds_obs = ds_obs.where(ds_obs.time >= start_datetime, drop=True)
    ds_obs = ds_obs.where(ds_obs.time <= end_datetime, drop=True)
    #
    ds_mdl = ensure_dataset_datetime64(ds_mdl)
    ds_mdl = ds_mdl.where(ds_mdl.time >= start_datetime, drop=True)
    ds_mdl = ds_mdl.where(ds_mdl.time <= end_datetime, drop=True)
    _logger.info('  -- time>=%s  &  time<=%s', start_datetime, end_datetime)

    _logger.info('Selected bounds for CMIP:')
    _logger.info('  -- model=%s', model_name)
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
    if station_or_globalmean == 'station':
        mdl_cell = get_closest_mdl_cell_dict(new_self.stepB_preprocessed_datasets[model_name],
                                             lat=obs_collection.station_dict[station_code]['lat'],
                                             lon=obs_collection.station_dict[station_code]['lon'],
                                             coords_as_dimensions=True)
        da_mdl = (ds_mdl
                  .where(ds_mdl.lat == mdl_cell['lat'], drop=True)
                  .where(ds_mdl.lon == mdl_cell['lon'], drop=True)
                  )
        _logger.info('  -- lat=%s', mdl_cell['lat'])
        _logger.info('  -- lon=%s', mdl_cell['lon'])
    elif station_or_globalmean == 'global':
        da_mdl = ds_mdl.mean(dim=('lat', 'lon'))
        _logger.info('  -- mean over lat and lon dimensions')
    else:
        raise ValueError(f'Unexpected value <{station_or_globalmean}> for station_or_globalmean parameter')
    # Lazy computations are executed.
    _logger.info('Applying selected bounds...')
    da_mdl = da_mdl.squeeze().compute()
    _logger.info('DONE.')

    if savepath_figure:
        # --- Create Graphic ---
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        if absolute_or_difference == 'difference':
            # Values at the same time
            da_mdl_rs = da_mdl.resample(time="1MS").mean()
            da_obs_rs = ds_obs.resample(time="1MS").mean()
            #
            da_TestMinusRef = (da_mdl_rs - da_obs_rs['co2']).dropna(dim='time')
            #
            # Plot
            ax.plot(da_TestMinusRef['time'], da_TestMinusRef, label='model - obs',
                    marker='.', linestyle='none')
            #
            ax.set_ylim(limits_with_zero(ax.get_ylim()))

        else:
            x_mdl = da_mdl['time'][~np.isnan(da_mdl.values)]
            y_mdl = da_mdl.values[~np.isnan(da_mdl.values)]
            #
            # Plot
            ax.plot(ds_obs['time'], ds_obs['co2'],
                    label=f'Obs [{station_code}]',
                    color='k')
            ax.plot(x_mdl, y_mdl,
                    label=f'Model [{model_name}]',
                    color='r', linestyle='-')

        ax.set_ylabel('$CO_2$ (ppm)')
        aesthetic_grid_no_spines(ax)
        #
        plt.legend()
        #
        plt.tight_layout()
        #
        mysavefig(fig=fig, plot_save_name=savepath_figure)

    return da_mdl
