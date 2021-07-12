from co2_diag.operations.time import ensure_dataset_datetime64
from co2_diag.operations.geographic import get_closest_mdl_cell_dict
from co2_diag.operations.utils import assert_expected_dimensions
from ccgcrv.ccg_dates import decimalDateFromDatetime
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from typing import Union
import logging

_logger = logging.getLogger(__name__)


def make_comparable(ref, com, **keywords):
    """Make two datasets comparable.

    Ensures time formats are compatible.
    Clips the data to appropriate time bounds.
    Gets data at the specified lat/lon.

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
    altitude_method : str
        either "interp" (provided with an altitude value) or "lowest" (default)
    altitude : float
        If altitude_method=='interp', altitude must be provided
    height_data : xarray.DataArray
        If altitude_method=='interp', height_data must be provided
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
    altitude_method = keywords.get("altitude_method", "lowest")
    altitude = keywords.get("altitude", None)
    height_data = keywords.get("height_data", None)
    global_mean = keywords.get("global_mean", False)
    verbose = keywords.get("verbose", "INFO")

    # Check the temporal domain of both
    # if ref.time != com.time:
    #     msg = "%s Datasets are not uniformly temporal: " % logstring
    #     msg += "reference = %s, comparison = %s" % (ref.temporal, com.temporal)
    #     logger.debug(msg)
    #     raise VarsNotComparable()

    _logger.info('Selected bounds for both:')

    # Apply time bounds to the reference, and then clip the comparison Dataset to the reference bounds.
    ds_ref, initial_ref_time, final_ref_time = apply_time_bounds(ref, time_limits)
    ds_com, initial_com_time, final_com_time = apply_time_bounds(com, (initial_ref_time, final_ref_time))
    # decimal years are added as a coordinate if not already there.
    if not ('time_decimal' in ds_com.coords):
        ds_com = ds_com.assign_coords(time_decimal=('time',
                                                    [decimalDateFromDatetime(x) for x in
                                                     pd.DatetimeIndex(ds_com['time'].values)]))
    _logger.info('  -- time>=%s  &  time<=%s', time_limits[0], time_limits[1])

    _logger.info('Selected bounds for Comparison dataset:')
    # _logger.info('  -- model=%s', opts.model_name)
    # Only the first ensemble member is selected, if there are more than one
    # (TODO: enable the selection of a specific ensemble member)
    if 'member_id' in ds_com['co2'].coords:
        ds_com = ds_com.isel(member_id=0)
        _logger.info('  -- member_id=0')
    if 'bnds' in ds_com['co2'].coords:
        ds_com = ds_com.isel(bnds=0, drop=True)

    assert_expected_dimensions(ds_com, expected_dims=['time', 'plev', 'lon', 'lat'], optional_dims=['bnds'])

    # A specific lat/lon is selected, or a global mean is calculated.
    # TODO: Add option for hemispheric averages as well.
    #  And average not only the CMIP model outputs the stations, but also the surface stations within that hemisphere.
    if global_mean:
        ds_com = ds_com.mean(dim=('lat', 'lon'))
        _logger.info('  -- mean over lat and lon dimensions')
    else:
        ds_com = extract_site_data_from_dataset(ds_com, lat=latlon[0], lon=latlon[1])

    assert_expected_dimensions(ds_com, expected_dims=['time', 'plev'], optional_dims=['bnds'])

    # Lazy computations are executed.
    _logger.info('Applying selected bounds...')
    ds_com = ds_com.compute()

    _logger.info('Extracting particular altitude...')
    da_com = extract_data_at_an_altitude(ds_com['co2'],
                                         altitude_method=altitude_method,
                                         altitude=altitude, height_data=ds_com['zg'])

    # Lazy computations are executed.
    da_com = da_com.squeeze()
    _logger.info('done.')

    return ds_ref, da_com


def extract_site_data_from_dataset(dataset: xr.Dataset,
                                   lat, lon) -> Union[xr.Dataset, xr.DataArray]:
    """A specific lat/lon is selected

    Parameters
    ----------
    dataset
    lat
    lon

    Raises
    ------
    ValueError, if an unexpected value for the method argument is given.

    Returns
    -------
    An xarray Dataset or DataArray
    """
    mdl_cell = get_closest_mdl_cell_dict(dataset, lat=lat, lon=lon, coords_as_dimensions=True)
    # data_subset = dataset.sel(lat=mdl_cell['lat'], lon=mdl_cell['lon'])
    data_subset = (dataset
                   .where(dataset.lat == mdl_cell['lat'], drop=True)
                   .where(dataset.lon == mdl_cell['lon'], drop=True)
                   .squeeze()
                   )
    _logger.info('  -- lat=%s', mdl_cell['lat'])
    _logger.info('  -- lon=%s', mdl_cell['lon'])

    return data_subset


def extract_data_at_an_altitude(data: xr.DataArray, altitude_method: str,
                                altitude: float, height_data: xr.DataArray = None
                                ) -> xr.DataArray:
    """Get values at a particular altitude
    If altitude_method is 'interp', then the 'altitude' and 'height_data' arguments are required.


    Parameters
    ----------
    data
    altitude_method
    altitude
    height_data

    Returns
    -------

    """
    if altitude_method == 'interp':
        data = interpolate_to_altitude(data=data, altitude=altitude, height_data=height_data)
    elif altitude_method == 'lowest':
        data = lowest_nonnull_altitude(data=data)
    else:
        raise ValueError('Unexpected altitude matching method, %s, for getting site data.'
                         % altitude_method)
    return data


def lowest_nonnull_altitude(data: xr.DataArray) -> xr.DataArray:
    """Get the lowest (i.e., first) non-null value, that is nearest to the surface

    Note: this assumes that data are ordered from the surface to top-of-atmosphere.
    """
    def first_nonnull_1d(data):
        # print(np.where(np.isfinite(data))[0][0])
        # print(np.isfinite(data))
        # print(data.plev[np.isfinite(data)])
        return data[np.isfinite(data)][0]

    print(data.plev)

    da_final = xr.apply_ufunc(
        first_nonnull_1d,  # first the function
        data,
        input_core_dims=[["plev"]],  # list with one entry per arg
        exclude_dims=set(("plev",)),  # dimensions allowed to change size. Must be set!
        vectorize=True)

    return da_final


def interpolate_to_altitude(data: xr.DataArray,
                            altitude: float,
                            height_data: xr.DataArray
                            ) -> xr.DataArray:
    """ Interpolate timeseries data to a given altitude

    Parameters
    ----------
    data: xarray.DataArray
        The carbon dioxide ('co2') variable
    altitude: float
    height_data: xarray.DataArray
        The geopotential height ('zg') variable.

    """
    if not all(x in data.data_vars for x in ['co2', 'zg']):
        raise ValueError("Variables 'co2' and 'zg' must be present to use interpolate_to_altitude()."
                         "Dataset only contains <%s>." % list(data.data_vars))

    def interp1d_np(data, x, xi):
        """
        data: y-coordinates of the data points (fp), e.g., array of plev
        x: x-coordinates of the data points (xp), e.g., array of zg
        xi: x-coordinate (e.g., zg) at which to evaluate an interpolated data point (e.g., plev)
        """
        return np.interp(xi, x, data)

    # For the given altitude (zg), a pressure level (plev) is interpolated at each time along the time dimension.
    da_plev_points = xr.apply_ufunc(
        interp1d_np,  # first the function
        data['plev'],
        height_data,
        altitude,
        input_core_dims=[["plev"], ["plev"], []],  # list with one entry per arg
        exclude_dims=set(("plev",)),  # dimensions allowed to change size. Must be set!
        vectorize=True)

    # For the given pressure level (plev) at each time,
    #   a concentration (co2) is interpolated at each time along the time dimension.
    da_final = xr.apply_ufunc(
        interp1d_np,  # first the function
        data['co2'],
        data['plev'],
        da_plev_points,
        input_core_dims=[["plev"], ["plev"], []],  # list with one entry per arg
        exclude_dims=set(("plev",)),  # dimensions allowed to change size. Must be set!
        vectorize=True)

    return da_final


def apply_time_bounds(ds: xr.Dataset,
                      time_limits: tuple
                      ) -> tuple:
    """

    Parameters
    ----------
    ds
    time_limits

    Returns
    -------
    a tuple containing:
        a Dataset
        the earliest datetime in the dataset
        the latest datetime in the dataset
    """
    ds = ensure_dataset_datetime64(ds)

    initial_ref_time = ds['time'].min().values
    final_ref_time = ds['time'].max().values

    if time_limits[0]:
        if final_ref_time < time_limits[0]:
            raise RuntimeError("Final time of dataset <%s> is before the given time frame's start <%s>." %
                               (np.datetime_as_string(final_ref_time, unit='s'), time_limits[0]))
        ds = ds.where(ds.time >= time_limits[0], drop=True)
    if time_limits[1]:
        if initial_ref_time > time_limits[1]:
            raise RuntimeError("Initial time of dataset <%s> is after the given time frame's end <%s>." %
                               (np.datetime_as_string(initial_ref_time, unit='s'), time_limits[1]))
        ds = ds.where(ds.time <= time_limits[1], drop=True)

    return ds, initial_ref_time, final_ref_time
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
