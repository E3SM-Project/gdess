# # import numpy as np
import pandas as pd
import xarray as xr
# #
from co2_diag.operations.time import ensure_dataset_datetime64, year_to_datetime64
from co2_diag.operations.geographic import get_closest_mdl_cell_dict
import co2_diag.data_source.obspack.surface_stations.collection as obspack_surface_collection_module

from ccgcrv.ccg_dates import datetimeFromDecimalDate, calendarDate, decimalDateFromDatetime

import logging
_logger = logging.getLogger(__name__)


def make_comparable(ref, com, **keywords):
    """Make two datasets comparable.

    Parameters
    ----------
    ref : xarray.Dataset
        the reference variable object
    com : xarray.Dataset
        the comparison variable object
    time_limits : tuple
        the start and end times
    latlon : tuple
        the latitude and longitude
    global_mean : bool
        whether to calculate the global mean instead of grabbing the nearest model location to the station
    verbose : Union[bool, str]
        e.g. "INFO", "DEBUG", or True

    Returns
    -------
    ref : xarray.Dataset
        the modified reference variable object
    com : xarray.Dataset
        the modified comparison variable object

    """

    # Process keywords
    time_limits = keywords.get("time_limits", (None, None))
    latlon = keywords.get("latlon", (None, None))
    global_mean = keywords.get("global_mean", False)
    verbose = keywords.get("verbose", "INFO")

    # Check the temporal domain of both
    # if ref.time != com.time:
    #     msg = "%s Datasets are not uniformly temporal: " % logstring
    #     msg += "reference = %s, comparison = %s" % (ref.temporal, com.temporal)
    #     logger.debug(msg)
    #     raise VarsNotComparable()

    _logger.info('Selected bounds for both:')

    # Ensure time formats are equivalent
    ds_ref = ensure_dataset_datetime64(ref)
    ds_com = ensure_dataset_datetime64(com)
    # Apply time bounds to the reference, and then clip the comparison Dataset to the reference bounds.
    if time_limits[0]:
        ds_ref = ds_ref.where(ds_ref.time >= time_limits[0], drop=True)
    if time_limits[1]:
        ds_ref = ds_ref.where(ds_ref.time <= time_limits[1], drop=True)
    ds_com = ds_com.where(ds_com.time >= ds_ref['time'].min().values, drop=True)
    ds_com = ds_com.where(ds_com.time <= ds_ref['time'].max().values, drop=True)
    _logger.info('  -- time>=%s  &  time<=%s', time_limits[0], time_limits[1])
    # decimal years are added as a coordinate if not already there.
    if not ('time_decimal' in ds_com.coords):
        ds_com = ds_com.assign_coords(time_decimal=('time',
                                                    [decimalDateFromDatetime(x) for x in
                                                     pd.DatetimeIndex(ds_com['time'].values)]))

    _logger.info('Selected bounds for Comparison dataset:')
    # _logger.info('  -- model=%s', opts.model_name)
    # Only the first ensemble member is selected, if there are more than one
    # (TODO: enable the selection of a specific ensemble member)
    if 'member_id' in ds_com['co2'].coords:
        ds_com = (ds_com
                  .isel(member_id=0)
                  .copy())
        _logger.info('  -- member_id=0')
    else:
        ds_com = ds_com.copy()

    # Surface values are selected.
    ds_com = ds_com['co2'].isel(plev=0)
    _logger.info('  -- plev=0')

    # A specific lat/lon is selected, or a global mean is calculated.
    # TODO: Add option for hemispheric averages as well.
    #  And average not only the CMIP model outputs the stations, but also the surface stations within that hemisphere.
    if global_mean:
        da_mdl = ds_com.mean(dim=('lat', 'lon'))
        _logger.info('  -- mean over lat and lon dimensions')
    else:
        obs_collection = obspack_surface_collection_module.Collection(verbose=verbose)
        mdl_cell = get_closest_mdl_cell_dict(ds_com,
                                             lat=latlon[0], lon=latlon[1],
                                             coords_as_dimensions=True)
        da_mdl = (ds_com
                  .where(ds_com.lat == mdl_cell['lat'], drop=True)
                  .where(ds_com.lon == mdl_cell['lon'], drop=True)
                  )
        _logger.info('  -- lat=%s', mdl_cell['lat'])
        _logger.info('  -- lon=%s', mdl_cell['lon'])

    # Lazy computations are executed.
    _logger.info('Applying selected bounds...')
    da_mdl = da_mdl.squeeze().compute()
    _logger.info('done.')

    return ds_ref, da_mdl
#
#
# # def get_combined_dataframe(dataset_ref: xr.Dataset,
# #                            dataset_e3sm: xr.Dataset,
# #                            timestart: np.datetime64,
# #                            timeend: np.datetime64,
# #                            ref_var: str = 'value',
# #                            e3sm_var: str = 'CO2'
# #                            ) -> pd.DataFrame:
# #     """Combine E3SM station data and a reference dataset
# #     into a pandas Dataframe for a specified time period
# #
# #     Parameters
# #     ----------
# #     dataset_ref
# #     dataset_e3sm
# #     timestart
# #     timeend
# #     ref_var
# #     e3sm_var
# #
# #     Returns
# #     -------
# #
# #     """
# #     # ----------------------
# #     # ----- REFERENCE ------
# #     # ----------------------
# #     ds_sub_ref = select_between(dataset=dataset_ref,
# #                                 timestart=timestart,
# #                                 timeend=timeend,
# #                                 varlist=['time', ref_var])
# #     new_ref_varname = 'ref_' + ref_var
# #     df_prepd_ref = (ds_sub_ref
# #                     .to_dataframe()
# #                     .reset_index()
# #                     .rename(columns={ref_var: new_ref_varname})
# #                     )
# #
# #     # # Get time-resampled resolution
# #     # ds_prepd_ref_resamp = ds_sub_ref.where(tempmask, drop=True).copy()
# #     # #     ds_prepd_ref_resamp = ds_prepd_ref_resamp.resample(time="1D").interpolate("linear")  # weekly average
# #     # ds_prepd_ref_resamp = ds_prepd_ref_resamp.resample(time="1MS").mean()  # monthly average
# #     # # ds_prepd = ds_sub.resample(time="1AS").mean()  # yearly average
# #     # # ds_prepd = ds_sub.resample(time="Q").mean()  # quarterly average (consecutive three-month periods)
# #     # # ds_prepd = ds_sub.resample(time="QS-DEC").mean()  # quarterly average (consecutive three-month periods), anchored at December 1st.
# #     # #
# #     # df_prepd_ref_resamp = (ds_prepd_ref_resamp
# #     #                        .dropna(dim=('time'))
# #     #                        .to_dataframe()
# #     #                        .reset_index()
# #     #                        .rename(columns={ref_var: 'ref_' + ref_var + '_resampled_resolution'})
# #     #                        )
# #
# #     # ----------------------
# #     # ---- E3SM Output -----
# #     # ----------------------
# #     ds_sub_e3sm = select_between(dataset=dataset_e3sm,
# #                                  timestart=timestart,
# #                                  timeend=timeend,
# #                                  varlist=['time', e3sm_var])
# #     new_e3sm_varname = 'e3sm_' + e3sm_var
# #     df_prepd_e3sm = (ds_sub_e3sm
# #                      .to_dataframe()
# #                      .reset_index()
# #                      .drop(columns=['ncol', 'lat', 'lon'])
# #                      .rename(columns={e3sm_var: new_e3sm_varname})
# #                      )
# #     # get the vertical mean
# #     #     ds_prepd_e3sm_orig = ds_prepd_e3sm_orig.mean(dim='lev').where(ds_mdl['ncol']==closest_mdl_point_dict['index'], drop=True)
# #
# #     # get the lowest level
# #     ds_prepd_e3sm_orig = (ds_prepd_e3sm_orig
# #                           .sel(lev=dataset_e3sm['lev'][-1])
# #                           .where(dataset_e3sm['ncol'] == closest_mdl_point_dict['index'], drop=True)
# #                           )
# #
# #     # Resample the time
# #     #     ds_prepd_e3sm_resamp = df_prepd_e3sm_orig.resample(time="1D").interpolate("linear")  # weekly average
# #
# #     # ------------------
# #     # ---- COMBINED ----
# #     # ------------------
# #     df_prepd = (df_prepd_ref
# #                 .merge(df_prepd_e3sm, on='time', how='outer')
# #                 .reset_index()
# #                 .loc[:, ['time', new_ref_varname, new_e3sm_varname]]
# #                 )
# #     # df_prepd['obs_original_resolution'] = df_prepd['obs_original_resolution'].astype(float)
# #     # df_prepd['obs_resampled_resolution'] = df_prepd['obs_resampled_resolution'].astype(float)
# #     # df_prepd['model_original_resolution'] = df_prepd['model_original_resolution'].astype(float)
# #     # df_prepd.rename(columns={'obs_original_resolution': 'NOAA Obs',
# #     #                          'obs_resampled_resolution': 'NOAA Obs daily mean',
# #     #                          'model_original_resolution': 'E3SM'}, inplace=True)
# #
# #     return df_prepd
