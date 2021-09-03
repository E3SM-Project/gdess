import co2_diag.graphics
from ccgcrv.ccg_filter import ccgFilter
from co2_diag import set_verbose
from co2_diag.data_source.models.cmip.cmip_collection import Collection as cmipCollection
from co2_diag.graphics.single_source_plots import plot_filter_components
from co2_diag.operations.time import ensure_dataset_datetime64, t2dt
from co2_diag.operations.geographic import get_closest_mdl_cell_dict
from co2_diag.operations.utils import assert_expected_dimensions
from co2_diag.formatters import append_before_extension
from co2_diag.data_source.observations import gvplus_surface as obspack_surface_collection_module
from ccgcrv.ccg_dates import decimalDateFromDatetime
from sklearn.metrics import mean_squared_error
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from typing import Union
import csv, sys, logging

_logger = logging.getLogger(__name__)


class Confrontation:
    def __init__(self,
                 compare_against_model: bool,
                 ds_mdl,
                 opts,
                 stations_to_analyze: list,
                 verbose: Union[bool, str] = False):
        """Instantiate a Confrontation object.

        Parameters
        ----------
        compare_against_model : bool
        ds_mdl : xarray Dataset
        opts : argparse.Namespace
        stations_to_analyze : list
        verbose : Union[bool, str], default False
        """
        self.compare_against_model = compare_against_model
        self.ds_mdl = ds_mdl
        self.opts = opts
        self.stations_to_analyze = stations_to_analyze
        self.verbose = verbose

        set_verbose(_logger, verbose)

    def looper(self, how):
        """

        Parameters
        ----------
        how : str
            either 'seasonal' or 'trend'

        Raises
        ------
        ValueError

        Returns
        -------
        tuple
            A bunch of things
        """
        valid = {'seasonal', 'trend'}
        if how not in valid:
            raise ValueError("'how' must be one of %r." % valid)

        # --- Observation data are processed for each station location. ---
        _logger.info('*Processing Observations*')
        counter = {'current': 1, 'skipped': 0}
        processed_station_metadata = dict(lat=[], lon=[], code=[], fullname=[])
        data_dict = dict(ref=[], mdl=[])  # each key will contain a list of Dataframes.
        num_stations = [len(self.stations_to_analyze)]
        for station in self.stations_to_analyze:
            _logger.info("Station %s of %s: %s", counter['current'], num_stations[0], station)
            obs_collection = obspack_surface_collection_module.Collection(verbose=self.verbose)
            obs_collection.preprocess(datadir=self.opts.ref_data, station_name=station)
            ds_obs = obs_collection.stepA_original_datasets[station]
            _logger.info('  %s', obs_collection.station_dict.get(station))

            # Apply time bounds, and get the relevant model output.
            try:
                if self.compare_against_model:
                    ds_obs, da_mdl = make_comparable(ds_obs, self.ds_mdl,
                                                     time_limits=(
                                                     np.datetime64(self.opts.start_yr), np.datetime64(self.opts.end_yr)),
                                                     latlon=(
                                                     ds_obs['latitude'].values[0], ds_obs['longitude'].values[0]),
                                                     altitude=ds_obs['altitude'].values[0], altitude_method='lowest',
                                                     global_mean=self.opts.globalmean, verbose=self.verbose)
                else:
                    ds_obs, _, _ = apply_time_bounds(ds_obs, time_limits=(np.datetime64(self.opts.start_yr),
                                                                          np.datetime64(self.opts.end_yr)))
                    da_mdl = None
            except (RuntimeError, AssertionError) as re:
                update_for_skipped_station(re, station, num_stations, counter)
                continue
            #
            if how == 'seasonal':
                ref_dt, ref_vals, mdl_dt, mdl_vals = get_seasonal_by_curve_fitting(self.compare_against_model,
                                                          da_mdl, ds_obs, self.opts, station)
                if isinstance(data_dict, Exception):
                    update_for_skipped_station(data_dict, station, num_stations, counter)
                    continue
                #
                data_dict['ref'].append(pd.DataFrame.from_dict({"month": ref_dt, f"{station}": ref_vals}))
                if self.compare_against_model:
                    data_dict['mdl'].append(pd.DataFrame.from_dict({"month": mdl_dt, f"{station}": mdl_vals}))
            elif how == 'trend':
                data_dict['ref'].append(pd.DataFrame.from_dict({"time": ds_obs['time'], f"{station}": ds_obs['co2'].values}))
                if self.compare_against_model:
                    data_dict['mdl'].append(pd.DataFrame.from_dict({"time": da_mdl['time'], f"{station}": da_mdl.values}))
            else:
                raise ValueError("Unexpected value for 'how' to do the Confrontation. Got %s." % how)

            # Gather together station's metadata at the loop end, when we're sure that this station has been processed.
            processed_station_metadata['lon'].append(obs_collection.station_dict[station]['lon'])
            processed_station_metadata['lat'].append(obs_collection.station_dict[station]['lat'])
            processed_station_metadata['fullname'].append(obs_collection.station_dict[station]['name'])
            processed_station_metadata['code'].append(station)
            counter['current'] += 1
            # END of station loop

        if len(data_dict['ref']) < 1:
            _logger.info("No station data to process (%s stations skipped). Exiting.", counter['skipped'])
            sys.exit()
        else:
            _logger.info("Done -- %s stations fully processed. %s stations skipped.",
                         len(data_dict['ref']), counter['skipped'])


        concatenated_dfs, df_station_metadata = self.concatenate_stations_and_months(data_dict,
                                                                                     processed_station_metadata)
        if how == 'seasonal':
            # concatenated_dfs, df_station_metadata = self.concatenate_stations_and_months(data_dict,
            #                                                                             processed_station_metadata)

            # --- Optional binning by latitude ---
            if self.opts.latitude_bin_size:
                concatenated_dfs, df_station_metadata = bin_by_latitude(self.compare_against_model, concatenated_dfs,
                                                                       df_station_metadata, self.opts.latitude_bin_size)

        # --- FORMAT DATA FOR OUTPUT ---

        # Write output data to csv
        filename = append_before_extension(self.opts.figure_savepath + '.csv',
                                           'seasonal_cycle_output_stats_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        fileptr = open(filename, 'w', newline='')
        writer = csv.DictWriter(
            fileptr, fieldnames=['station',
                                 'source',
                                 'max',
                                 'min',
                                 'mean',
                                 'median',
                                 'std',
                                 'rmse'
                                 ]
        )
        writer.writeheader()

        if how == 'seasonal':
            xdata_gv = concatenated_dfs['ref']['month']
            ydata_gv = concatenated_dfs['ref'].loc[:, (concatenated_dfs['ref'].columns != 'month')]
        elif how == 'trend':
            xdata_gv = concatenated_dfs['ref']['time']
            ydata_gv = concatenated_dfs['ref'].loc[:, (concatenated_dfs['ref'].columns != 'time')]
        else:
            raise ValueError("Unexpected value for 'how' to do the Confrontation. Got %s." % how)

        # Write output data for this instance
        for column in ydata_gv:
            row_dict = {
                'station': column,
                'source': 'globalviewplus',
                'max': ydata_gv[column].max(),
                'min': ydata_gv[column].min(),
                'mean': ydata_gv[column].mean(),
                'median': ydata_gv[column].median(),
                'std': ydata_gv[column].std(),
                'rmse': np.nan
            }
            writer.writerow(row_dict)

        xdata_mdl = None
        ydata_mdl = None
        if self.compare_against_model:
            if how == 'seasonal':
                xdata_mdl = concatenated_dfs['mdl']['month']
                if not xdata_gv.equals(xdata_mdl):
                    raise ValueError(
                        'Unexpected discrepancy, xdata for reference observations does not equal xdata for models')
                ydata_mdl = concatenated_dfs['mdl'].loc[:, (concatenated_dfs['mdl'].columns != 'month')]
                rmse_y_true = ydata_gv
                rmse_y_pred = ydata_mdl

            elif how == 'trend':
                xdata_mdl = concatenated_dfs['mdl']['time']

                begin_time_for_stats = max(xdata_gv.min(), xdata_mdl.min())
                end_time_for_stats = min(xdata_gv.max(), xdata_mdl.max())
                if begin_time_for_stats > end_time_for_stats:
                    _logger.info('beginning time <%s> is after end time <%s>' %
                                 (begin_time_for_stats, end_time_for_stats))
                    rmse_y_true = None
                    rmse_y_pred = None
                else:
                    def month_calc(df):
                        return (df
                                .where((df['time'] < end_time_for_stats) & (df['time'] > begin_time_for_stats))
                                .dropna(subset=['time'], how='any', inplace=False)
                                .resample("1MS", on='time')
                                .mean()
                                .reset_index())
                    rmse_y_true = month_calc(concatenated_dfs['ref'])
                    rmse_y_pred = month_calc(concatenated_dfs['mdl'])
                    common_time = set(rmse_y_true['time']).intersection(set(rmse_y_pred['time']))
                    rmse_y_true = rmse_y_true.loc[rmse_y_true['time'].isin(common_time), :]
                    rmse_y_pred = rmse_y_pred.loc[rmse_y_pred['time'].isin(common_time), :]

                ydata_mdl = concatenated_dfs['mdl'].loc[:, (concatenated_dfs['mdl'].columns != 'time')]
            else:
                raise ValueError("Unexpected value for 'how' to do the Confrontation. Got %s." % how)

            rmse = np.nan
            if rmse_y_true is not None:
                yt = rmse_y_true[column]
                yp = rmse_y_pred[column]
                okayvals = yt.notnull() & yp.notnull()
                rmse = mean_squared_error(yt[okayvals], yp[okayvals], squared=False)

            # Write output data for this instance
            for column in ydata_mdl:
                row_dict = {
                    'station': column,
                    'source': 'cmip',
                    'max': ydata_mdl[column].max(),
                    'min': ydata_mdl[column].min(),
                    'mean': ydata_mdl[column].mean(),
                    'median': ydata_mdl[column].median(),
                    'std': ydata_mdl[column].std(),
                    'rmse': rmse
                }
                writer.writerow(row_dict)
        fileptr.flush()

        return data_dict, concatenated_dfs, df_station_metadata, xdata_gv, xdata_mdl, ydata_gv, ydata_mdl

    def concatenate_stations_and_months(self, data_dict, processed_station_metadata) -> (dict, pd.DataFrame):
        """

        Parameters
        ----------
        data_dict : dict
            each key contains a list of Dataframes
        processed_station_metadata

        Returns
        -------
        dict
            A dictionary with two dataframes, in which each column is a different station.
        pd.Dataframe
            metadata for all stations.
        """
        # Dataframes for each location are combined so we have one 'month' column, and a single column for each station.
        # First, dataframes are sorted by latitude, then combined, then the duplicate 'month' columns are removed.
        df_station_metadata = pd.DataFrame.from_dict(processed_station_metadata)
        df_concatenated = dict(ref=None, mdl=None)

        #   (i) Globalview+ data
        data_dict['ref'] = [x for _, x in sorted(zip(list(df_station_metadata['lat']), data_dict['ref']))]
        df_concatenated['ref'] = pd.concat(data_dict['ref'], axis=1, sort=False)
        df_concatenated['ref'] = df_concatenated['ref'].loc[:, ~df_concatenated['ref'].columns.duplicated()]

        #   (ii) CMIP data
        if self.compare_against_model:
            data_dict['mdl'] = [x for _, x in sorted(zip(list(df_station_metadata['lat']), data_dict['mdl']))]
            df_concatenated['mdl'] = pd.concat(data_dict['mdl'], axis=1, sort=False)
            df_concatenated['mdl'] = df_concatenated['mdl'].loc[:, ~df_concatenated['mdl'].columns.duplicated()]
        #
        # Sort the metadata after using it for sorting the cycle list(s)
        df_station_metadata.sort_values(by='lat', ascending=True, inplace=True)

        return df_concatenated, df_station_metadata


def make_comparable(ref: xr.Dataset, com: xr.Dataset, **keywords) -> (xr.Dataset, xr.DataArray):
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
    com : xarray.Dataarray
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

    if verbose:
        ProgressBar().register()

    # Check the temporal domain of both
    # if ref.time != com.time:
    #     msg = "%s Datasets are not uniformly temporal: " % logstring
    #     msg += "reference = %s, comparison = %s" % (ref.temporal, com.temporal)
    #     logger.debug(msg)
    #     raise VarsNotComparable()

    _logger.info('Selected bounds for both:')
    ds_com, ds_ref = mutual_time_bounds(com, ref, time_limits)

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
        ds_com = extract_site_data_from_dataset(ds_com, lat=latlon[0], lon=latlon[1], drop=True)

    assert_expected_dimensions(ds_com, expected_dims=['time', 'plev'], optional_dims=['bnds'])

    # Lazy computations are executed.
    _logger.info('Applying selected bounds...')
    ds_com = ds_com.compute()

    if altitude_method == 'interp':
        da_com = interpolate_to_altitude(data=ds_com['co2'], altitude=altitude, height_data=ds_com['zg'])
    elif altitude_method == 'lowest':
        da_com = lowest_nonnull_altitude(data=ds_com['co2'])
    else:
        raise ValueError('Unexpected altitude matching method, %s, for getting site data.'
                         % altitude_method)

    # Lazy computations are executed.
    da_com = da_com.squeeze()
    _logger.info('done.')

    return ds_ref, da_com


def mutual_time_bounds(com: xr.Dataset, ref: xr.Dataset, time_limits) -> (xr.Dataset, xr.Dataset):
    """

    Parameters
    ----------
    com : xr.Dataset
    ref : xr.Dataset
    time_limits : tuple of datetime
        (start time, end time)

    Returns
    -------
    tuple
        ds_com : xr.Dataset
        ds_ref : xr.Dataset
    """
    # Apply time bounds to the reference, and then clip the comparison Dataset to the reference bounds.
    ds_ref, initial_ref_time, final_ref_time = apply_time_bounds(ref, time_limits)
    ds_com, initial_com_time, final_com_time = apply_time_bounds(com, (ds_ref['time'].min().values, ds_ref['time'].max().values))
    # decimal years are added as a coordinate if not already there.
    if not ('time_decimal' in ds_com.coords):
        ds_com = ds_com.assign_coords(time_decimal=('time',
                                                    [decimalDateFromDatetime(x) for x in
                                                     pd.DatetimeIndex(ds_com['time'].values)]))
    _logger.info('  -- time>=%s  &  time<=%s', time_limits[0], time_limits[1])
    return ds_com, ds_ref


def extract_site_data_from_dataset(dataset: xr.Dataset,
                                   lat: float, lon: float,
                                   drop: bool) -> Union[xr.Dataset, xr.DataArray]:
    """A specific lat/lon is selected

    Parameters
    ----------
    dataset : xarray.Dataset
    lat : float
    lon : float
    drop : bool

    Raises
    ------
    ValueError, if an unexpected value for the method argument is given.

    Returns
    -------
    An xarray Dataset or DataArray
    """
    mdl_cell = get_closest_mdl_cell_dict(dataset, lat=lat, lon=lon, coords_as_dimensions=True)
    data_subset = dataset.sel({'lat': mdl_cell['lat'],
                               'lon': mdl_cell['lon']},
                              drop=drop)

    _logger.info('  -- lat=%s', mdl_cell['lat'])
    _logger.info('  -- lon=%s', mdl_cell['lon'])

    return data_subset


def lowest_nonnull_altitude(data: xr.DataArray) -> xr.DataArray:
    """Get the lowest (i.e., first) non-null value, that is nearest to the surface

    Note: this assumes that data are ordered from the surface to top-of-atmosphere.
    """
    def first_nonnull_1d(data):
        # print(np.where(np.isfinite(data))[0][0])
        # print(np.isfinite(data))
        # print(data.plev[np.isfinite(data)])
        return data[np.isfinite(data)][0]

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
    data : xarray.DataArray
        The carbon dioxide ('co2') variable
    altitude : float
    height_data : xarray.DataArray
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
                      ) -> (xr.Dataset, np.datetime64, np.datetime64):
    """

    Parameters
    ----------
    ds : xr.Dataset
    time_limits : tuple of datetime or list with length 2
        (start time, end time)

    Returns
    -------
    ds : xr.Dataset
    initial_time : xr.Dataset
        the earliest datetime in the dataset
    final_time : xr.Dataset
        the latest datetime in the dataset
    """
    ds = ensure_dataset_datetime64(ds)

    initial_time = ds['time'].min().values
    final_time = ds['time'].max().values

    if time_limits[0]:
        if final_time < time_limits[0]:
            raise RuntimeError("Final time of dataset <%s> is before the given time frame's start <%s>." %
                               (np.datetime_as_string(final_time, unit='s'), time_limits[0]))
        ds = ds.where(ds.time >= time_limits[0], drop=True)
    if time_limits[1]:
        if initial_time > time_limits[1]:
            raise RuntimeError("Initial time of dataset <%s> is after the given time frame's end <%s>." %
                               (np.datetime_as_string(initial_time, unit='s'), time_limits[1]))
        ds = ds.where(ds.time <= time_limits[1], drop=True)

    return ds, initial_time, final_time


def update_for_skipped_station(msg, station_name, station_count, counter_dict):
    """Print a message and reduce the total station count by one."""
    _logger.info('  skipping station <%s>: %s', station_name, msg)
    counter_dict['skipped'] += 1
    station_count[0] -= 1


def load_cmip_model_output(model_name: str,
                           cmip_load_method: str,
                           verbose=True) -> (bool, xr.Dataset):
    """Load CMIP model output

    We will only compare against CMIP model outputs if a model_name is supplied, otherwise return dataset as None.

    Parameters
    ----------
    model_name : str
    cmip_load_method : str
    verbose : bool, default True

    Returns
    -------
    bool
    xarray.Dataset
    """
    if compare_against_model := bool(model_name):
        _logger.info('*Processing CMIP model output*')
        new_self, _ = cmipCollection._recipe_base(datastore='cmip6', verbose=verbose, model_name=model_name,
                                                  load_method=cmip_load_method, skip_selections=True,
                                                  pickle_file=None)
        ds_mdl = new_self.stepB_preprocessed_datasets[model_name]
        ds_mdl = ds_mdl.assign_coords(time_decimal=('time', [decimalDateFromDatetime(x)
                                                             for x in pd.DatetimeIndex(ds_mdl['time'].values)]))
    else:
        ds_mdl = None
    return compare_against_model, ds_mdl


def bin_by_latitude(compare_against_model: bool,
                    data_dict: dict,
                    df_metadata: pd.DataFrame,
                    latitude_bin_size: int
                    ) -> tuple:
    """

    Parameters
    ----------
    compare_against_model : bool
    data_dict : dict
        each key contains a list of Dataframes
    df_metadata : pandas.Dataframe
    latitude_bin_size : int

    Returns
    -------
    dict
    pandas.Dataframe
    """
    # We determine bins to which each station is assigned.
    def to_bin(x):
        return np.floor(x / latitude_bin_size) * latitude_bin_size

    df_metadata["latbin"] = df_metadata['lat'].map(to_bin)
    df_metadata["lonbin"] = df_metadata['lon'].map(to_bin)
    #
    data_dict['ref'] = calc_binned_means(data_dict['ref'], df_metadata)
    if compare_against_model:
        data_dict['mdl'] = calc_binned_means(data_dict['mdl'], df_metadata)

    return data_dict, df_metadata


def get_seasonal_by_curve_fitting(compare_against_model: bool,
                                  da_mdl, ds_obs,
                                  opts, station):
    """

    Parameters
    ----------
    compare_against_model : bool
    da_mdl : xarray.Dataarray
    da_obs : xarray.Dataarray
    ds_obs : xarray.Dataset
    opts : argparse.Namespace
    station : str

    Raises
    ------
    ValueError

    Returns
    -------
    tuple
    """
    # Check that there is at least one year's worth of data for this station.
    if (ds_obs.time.values.max().astype('datetime64[M]') - ds_obs.time.values.min().astype('datetime64[M]')) < 12:
        _logger.info('  insufficient number of months of data for station <%s>' % station)
        return ValueError

    # --- Curve fitting ---
    #   (i) Globalview+ data
    filt_ref = ccgFilter(xp=ds_obs['time_decimal'].values, yp=ds_obs['co2'].values,
                         numpolyterms=3, numharmonics=4, timezero=int(ds_obs['time_decimal'].values[0]))
    #   (ii) CMIP data
    if compare_against_model:
        try:
            filt_mdl = ccgFilter(xp=da_mdl['time_decimal'].values, yp=da_mdl.values,
                                 numpolyterms=3, numharmonics=4, timezero=int(da_mdl['time_decimal'].values[0]))
        except TypeError as te:
            _logger.info('--- Curve filtering error ---')
            return te

    # Optional plotting of components of the filtering process
    if co2_diag.graphics.single_source_plots.plot_filter_components:
        plot_filter_components(filt_ref,
                               original_x=ds_obs['time_decimal'].values,
                               # df_surface_station['time_decimal'].values,
                               original_y=ds_obs['co2'].values,  # df_surface_station['co2'].values,
                               figure_title=f'obs, station {station}',
                               savepath=append_before_extension(opts.figure_savepath, 'supplement1ref_' + station))
        if compare_against_model:
            plot_filter_components(filt_mdl,
                                   original_x=da_mdl['time_decimal'].values,
                                   original_y=da_mdl.values,
                                   figure_title=f'model [{opts.model_name}]',
                                   savepath=append_before_extension(opts.figure_savepath, 'supplement1_mdl'))

    # --- Compute the annual climatological cycle ---
    #   (i) Globalview+ data
    ref_dt, ref_vals = make_cycle(x0=filt_ref.xinterp,
                                  smooth_cycle=filt_ref.getHarmonicValue(
                                      filt_ref.xinterp) + filt_ref.smooth - filt_ref.trend)
    #   (ii) CMIP data
    mdl_dt, mdl_vals = None, None
    if compare_against_model:
        mdl_dt, mdl_vals = make_cycle(x0=filt_mdl.xinterp,
                                      smooth_cycle=filt_mdl.getHarmonicValue(
                                          filt_mdl.xinterp) + filt_mdl.smooth - filt_mdl.trend)

    return ref_dt, ref_vals, mdl_dt, mdl_vals


def calc_binned_means(df_cycles_for_all_stations_ref: pd.DataFrame,
                      df_station_metadata: pd.DataFrame
                      ) -> pd.DataFrame:
    """Calculate means for each bin

    Note, this function expects a DataFrame column titled "latbin" designating bin assignments.

    Parameters
    ----------
    df_cycles_for_all_stations_ref : pandas.Dataframe
    df_station_metadata : pandas.Dataframe

    Returns
    -------
    pandas.Dataframe
    """
    # Add the coordinates and binning information to the dataframe with seasonal cycle values
    new_df = df_cycles_for_all_stations_ref.transpose()
    new_df.columns = new_df.loc['month']  # .map(lambda x: x.strftime('%m'))
    new_df = (new_df
              .drop(labels='month', axis=0, inplace=False)
              .apply(pd.to_numeric, axis=0)
              .reset_index()
              .rename(columns={'index': 'code'})
              .merge(df_station_metadata.loc[:, ['code', 'fullname', 'lat', 'latbin']], on='code'))

    # Take the means of each latitude bin and transpose dataframe
    groups = new_df.groupby(["latbin"], as_index=False)
    binned_df = (groups.mean()
                 .drop('lat', axis=1)
                 .sort_values(by='latbin', ascending=True)
                 .set_index('latbin')
                 .transpose()
                 .reset_index()
                 .rename(columns={'index': 'month'}))
    return binned_df


def make_cycle(x0, smooth_cycle) -> (pd.Series, pd.Series):
    """Calculate the average seasonal cycle from the filtered time series.

    Parameters
    ----------
    x0
    smooth_cycle

    Returns
    -------
    a tuple containing two pandas.Series of 12 elemenets: one of datetimes for each month, and one of co2 values
    """
    # Convert dates to datetime objects, and make a dataframe with a month column for grouping purposes.
    df_seasonalcycle = pd.DataFrame.from_dict({'datetime': [t2dt(i) for i in x0],
                                               'co2': smooth_cycle})
    df_seasonalcycle['month'] = df_seasonalcycle['datetime'].dt.month

    # Bin by month, and add a column that represents months in datetime format for plotting purposes.
    df_monthly = df_seasonalcycle.groupby('month').mean().reset_index()
    df_monthly['month_datetime'] = pd.to_datetime(df_monthly['month'], format='%m')

    return df_monthly['month_datetime'], df_monthly['co2']