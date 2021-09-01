import co2_diag.graphics.single_source_plots
from co2_diag import load_stations_dict
from co2_diag.data_source.models.cmip.cmip_collection import Collection as cmipCollection
from co2_diag.formatters import append_before_extension
from co2_diag.operations.time import t2dt
from co2_diag.graphics.single_source_plots import plot_filter_components
from ccgcrv.ccg_dates import decimalDateFromDatetime
from ccgcrv.ccg_filter import ccgFilter
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union
import logging

_logger = logging.getLogger(__name__)


def load_cmip_model_output(model_name: str,
                           cmip_load_method: str,
                           verbose=True) -> (bool, xr.Dataset):
    """Load CMIP model output

    We will only compare against CMIP model outputs if a model_name is supplied, otherwise return dataset as None.

    Parameters
    ----------
    model_name
    cmip_load_method
    verbose

    Returns
    -------

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


def populate_station_list(run_all_stations: bool,
                          station_list: Union[bool, list, str]) -> list:
    """The list of stations to analyze is populated.

    Parameters
    ----------
    run_all_stations
    station_list

    Returns
    -------

    """
    if run_all_stations:
        stations_dict = load_stations_dict()
        stations_to_analyze = stations_dict.keys()
    elif station_list:
        stations_to_analyze = station_list
    else:
        raise ValueError('Unexpected empty station list')

    return stations_to_analyze


def get_seasonal_by_curve_fitting(compare_against_model, data_dict, da_mdl, ds_obs,
                                  opts, station):
    """

    Parameters
    ----------
    compare_against_model: (bool)
    data_dict: (dict) each key contains a list of Dataframes
    da_mdl
    da_obs
    ds_obs
    opts
    station

    Returns
    -------

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


def bin_by_latitude(compare_against_model, data_dict, df_metadata, latitude_bin_size):
    """
    
    Parameters
    ----------
    compare_against_model
    data_dict
    df_metadata
    latitude_bin_size

    Returns
    -------

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


def calc_binned_means(df_cycles_for_all_stations_ref: pd.DataFrame, df_station_metadata: pd.DataFrame
                      ) -> pd.DataFrame:
    """Calculate means for each bin

    Note, this function expects a DataFrame column titled "latbin" designating bin assignments.

    Parameters
    ----------
    df_cycles_for_all_stations_ref
    df_station_metadata

    Returns
    -------

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


def update_for_skipped_station(msg, station_name, station_count, counter_dict):
    """Print a message and reduce the total station count by one."""
    _logger.info('  skipping station <%s>: %s', station_name, msg)
    counter_dict['skipped'] += 1
    station_count[0] -= 1


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
