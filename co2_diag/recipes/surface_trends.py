""" This produces a plot of multidecadal trends of atmospheric CO2
This function parses:
 - observational data from Globalview+ surface stations
 - model output from CMIP6
================================================================================
"""

import numpy as np

from co2_diag.recipes.utils import get_recipe_param
import co2_diag.dataset_operations.obspack.surface_stations.collection as obspack_surface_collection_module
import co2_diag.dataset_operations.cmip.collection as cmip_collection_module
from co2_diag.dataset_operations.geographic import get_closest_mdl_cell_dict

import matplotlib.pyplot as plt

from dask.diagnostics import ProgressBar

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
    # --- Recipe parameters are parsed. ---
    _logger.debug("Parsing additional parameters ---")
    ref_data = get_recipe_param(param_kw, 'ref_data', default_value=None)
    model_name = get_recipe_param(param_kw, 'model_name', default_value='GFDL')
    start_yr = get_recipe_param(param_kw, 'start_yr', default_value="1960")
    end_yr = get_recipe_param(param_kw, 'end_yr', default_value="2015")
    savepath_figure = get_recipe_param(param_kw, 'savepath_figure', default_value=None)
    station_or_globalmean = get_recipe_param(param_kw, 'station_or_globalmean', default_value='station')
    # For a single station, we also check that it is accounted for in the class attribute dict.
    station_code = get_recipe_param(param_kw, 'station_code', default_value='mlo')
    if station_code not in obspack_surface_collection_module.station_dict:
        raise ValueError('Unexpected station name <%s>', param_kw['stationshortname'])

    # --- Surface observations ---
    _logger.info('*Observations*')
    obs_collection = obspack_surface_collection_module.Collection(verbose=verbose)
    obs_collection.preprocess(datadir=ref_data, station_name=station_code)
    # Data are resampled
    obs_collection.df_combined_and_resampled = (obs_collection
                                                .get_resampled_dataframe(obs_collection.stepA_original_datasets[station_code],
                                                                         timestart=np.datetime64(start_yr),
                                                                         timeend=np.datetime64(end_yr)
                                                                         ).reset_index()
                                                )

    # --- CMIP model output at surface ---
    _logger.info('*CMIP model output*')
    cmip_collection = cmip_collection_module.Collection(verbose=verbose)
    new_self, loaded_from_file = cmip_collection._cmip_recipe_base(datastore='cmip6', verbose=verbose,
                                                                   load_from_file=None)
    ds = new_self.stepB_preprocessed_datasets[model_name]

    # --- Obspack and CMIP are now handled Together ---
    _logger.info('Applying selected bounds to CMIP..')
    if verbose:
        ProgressBar().register()
    # Surface values are selected.
    ds_surface = ds['co2'].isel(plev=0)
    _logger.info('  -- plev=0')
    # Only the first ensemble member is selected, if there are more than one
    # (TODO: enable the selection of a specific ensemble member)
    if 'member_id' in ds['co2'].coords:
        ds_onemember = (ds_surface
                        .isel(member_id=0)
                        .copy())
        _logger.info('  -- member_id=0')
    else:
        ds_onemember = ds_surface.copy()
    # A specific lat/lon is selected, or a global mean is calculated.
    if station_or_globalmean == 'station':
        mdl_cell = get_closest_mdl_cell_dict(new_self.stepB_preprocessed_datasets[model_name],
                                             lat=obs_collection.station_dict[station_code]['lat'],
                                             lon=obs_collection.station_dict[station_code]['lon'],
                                             coords_as_dimensions=True)
        da = (ds_onemember
              .where(ds.lat == mdl_cell['lat'], drop=True)
              .where(ds.lon == mdl_cell['lon'], drop=True)
              )
        _logger.info('  -- lat=%s', mdl_cell['lat'])
        _logger.info('  -- lon=%s', mdl_cell['lon'])
    elif station_or_globalmean == 'global':
        da = ds_onemember.mean(dim=('lat', 'lon'))
        _logger.info('  -- mean over lat and lon dimensions')
    else:
        raise ValueError(f'Unexpected value <{station_or_globalmean}> for station_or_globalmean parameter')
    # Lazy computations are executed.
    _logger.info('Applying selected bounds..')
    da = da.squeeze().compute()

    # --- Create Graphic ---
    if savepath_figure:
        dt_time = np.array([np.datetime64(item) for item in da['time'].values])
        x_mdl = dt_time[~np.isnan(da.values)]
        y_mdl = da.values[~np.isnan(da.values)]

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        #
        ax.plot(obs_collection.stepA_original_datasets[station_code]['time'],
                obs_collection.stepA_original_datasets[station_code]['co2'],
                label=f'Obs [{station_code}]',
                color='k')
        ax.plot(x_mdl, y_mdl,
                label='model', color='r', linestyle='-')
        #
        ax.set_xlim(np.datetime64('1980'), np.datetime64('2021'))
        ax.set_ylabel('$CO_2$ (ppm)')
        #
        plt.legend()
        #
        plt.tight_layout()
        plt.savefig(savepath_figure)

    return da
