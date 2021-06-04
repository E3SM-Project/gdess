""" This produces a plot of multidecadal trends of atmospheric CO2
This function parses:
 - observational data from Globalview+ surface stations
 - model output from CMIP6
================================================================================
"""
import argparse
import datetime
from datetime import timedelta
from typing import Union
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker

from co2_diag import set_verbose
import co2_diag.data_source.obspack.surface_stations.collection as obspack_surface_collection_module
import co2_diag.data_source.cmip.collection as cmip_collection_module
from co2_diag.formatters.nums import numstr
from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig, limits_with_zero
from co2_diag.recipes.utils import add_shared_arguments_for_recipes, parse_recipe_options

from co2_diag.operations.Confrontation import make_comparable, apply_time_bounds

from ccgcrv.ccg_filter import ccgFilter
from ccgcrv.ccg_dates import datetimeFromDecimalDate, calendarDate, decimalDateFromDatetime

import logging
_logger = logging.getLogger(__name__)


def seasonal_cycles(options: Union[dict, argparse.Namespace],
                    verbose: Union[bool, str] = False,
                    ):
    """Execute a series of preprocessing steps and generate a diagnostic result.

    Relevant co2_diag collections are instantiated and processed.

    If one station is specified, then that will be compared against model data at the same location
    If more than one station is specified, then no model data will be compared against it.

    Parameters
    ----------
    options: Union[dict, argparse.Namespace]
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
    opts = parse_recipe_options(options, add_seasonal_cycle_args_to_parser)

    # The list of stations to analyze is populated.
    if opts.run_all_stations:
        stations_to_analyze = obspack_surface_collection_module.station_dict.keys()
    elif opts.run_three_stations:
        stations_to_analyze = list(obspack_surface_collection_module.station_dict.keys())[0:3]
    elif opts.station_list:
        stations_to_analyze = opts.station_list
    else:
        stations_to_analyze = [opts.station_code]

    # --- Load CMIP model output ---
    # We will only compare against CMIP model outputs if we are analyzing the cycle for a single station
    ds_mdl = None
    compare_against_model = len(stations_to_analyze) < 2
    if compare_against_model:
        _logger.info('*Processing CMIP model output*')
        cmip_collection = cmip_collection_module.Collection(verbose=verbose)
        new_self, loaded_from_file = cmip_collection._recipe_base(datastore='cmip6', verbose=verbose,
                                                                  from_file=None, skip_selections=True)
        ds_mdl = new_self.stepB_preprocessed_datasets[opts.model_name]
        ds_mdl = ds_mdl.assign_coords(time_decimal=('time', [decimalDateFromDatetime(x)
                                                             for x in pd.DatetimeIndex(ds_mdl['time'].values)]))

    # --- Load Surface station observations ---
    _logger.info('*Processing Observations*')
    counter = {'current': 1, 'skipped': 0}
    station_filts = []
    processed_station_metadata = {'lat': [], 'lon': [], 'code': [], 'fullname': []}
    cycle_list_with_each_station = []
    num_stations = [len(stations_to_analyze)]
    for station in stations_to_analyze:
        _logger.info("Station %s of %s: %s", counter['current'], num_stations[0], station)
        obs_collection = obspack_surface_collection_module.Collection(verbose=verbose)
        obs_collection.preprocess(datadir=opts.ref_data, station_name=station)
        ds_obs = obs_collection.stepA_original_datasets[station]
        _logger.info('  %s', obs_collection.station_dict[station])

        # if opts.use_mlo_for_detrending:
        #     ds_mlo_ref = obs_collection.stepA_original_datasets['mlo']

        # Apply time bounds, and get the relevant model output.
        try:
            if compare_against_model:
                da_obs, da_mdl = make_comparable(ds_obs, ds_mdl,
                                                 time_limits=(np.datetime64(opts.start_yr),
                                                              np.datetime64(opts.end_yr)),
                                                 latlon=(ds_obs['latitude'].values[0], ds_obs['longitude'].values[0]),
                                                 global_mean=False)
            else:
                da_obs, _, _ = apply_time_bounds(ds_obs, time_limits=(np.datetime64(opts.start_yr),
                                                                      np.datetime64(opts.end_yr)))
        except RuntimeError as re:
            update_for_skipped_station(re, station, num_stations, counter)
            continue

        # Check that there is at least one year's worth of data for this station.
        num_months = da_obs.time.values.max().astype('datetime64[M]') - da_obs.time.values.min().astype('datetime64[M]')
        if num_months < 12:
            msg = '  insufficient number of months of data for station <%s>' % station
            update_for_skipped_station(msg, station, num_stations, counter)
            continue

        # --- Curve fitting ---
        # CMIP data at the location of the station
        if compare_against_model:
            xp = da_mdl['time_decimal'].values
            yp = da_mdl.values
            try:
                filt_mdl = ccgFilter(xp=xp, yp=yp, numpolyterms=3, numharmonics=4, timezero=int(xp[0]))
            except TypeError as te:
                _logger.info('--- Curve filtering error ---')
                update_for_skipped_station(te, station, num_stations, counter)
                continue
        # Surface stations
        xp = da_obs['time_decimal'].values
        yp = da_obs['co2'].values
        filt_ref = ccgFilter(xp=xp, yp=yp, numpolyterms=3, numharmonics=4, timezero=int(xp[0]))
        #
        station_filts.append(filt_ref)

        if opts.plot_filter_components:
            plot_filter_components(filt_ref,
                                   original_x=ds_obs['time_decimal'].values, #df_surface_station['time_decimal'].values,
                                   original_y=ds_obs['co2'].values,  #df_surface_station['co2'].values,
                                   figure_title=f'obs, station {station}',
                                   savepath=opts.figure_savepath + '_supplement1ref_' + station + '.png')

        # --- Compute the annual climatological cycle ---
        ref_dt, ref_vals = make_cycle(x0=filt_ref.xinterp,
                                     smooth_cycle=filt_ref.getHarmonicValue(filt_ref.xinterp) + filt_ref.smooth - filt_ref.trend)
        #
        cycle_list_with_each_station.append(pd.DataFrame.from_dict({"month": ref_dt, f"{station}": ref_vals}))

        # Gather together the metadata for this station now that it's been processed.
        processed_station_metadata['lon'].append(obs_collection.station_dict[station]['lon'])
        processed_station_metadata['lat'].append(obs_collection.station_dict[station]['lat'])
        processed_station_metadata['fullname'].append(obs_collection.station_dict[station]['name'])
        processed_station_metadata['code'].append(station)

        counter['current'] += 1
        # END of station loop

    _logger.info("Done -- %s stations fully processed. %s stations skipped.",
                 len(cycle_list_with_each_station), counter['skipped'])

    if compare_against_model:
        mdl_dt, mdl_vals = make_cycle(x0=filt_mdl.xinterp,
                   smooth_cycle=filt_mdl.getHarmonicValue(filt_mdl.xinterp) + filt_mdl.smooth - filt_mdl.trend)
        if opts.plot_filter_components:
            plot_filter_components(filt_mdl,
                                   original_x=da_mdl['time_decimal'].values,
                                   original_y=da_mdl.values,
                                   figure_title=f'model [{opts.model_name}]',
                                   savepath=opts.figure_savepath + 'supplement1_mdl.png')

    df_station_metadata = pd.DataFrame.from_dict(processed_station_metadata)
    # Dataframes for each station are combined so we have one 'month' column, and a single column for each station.
    # First, dataframes are sorted by latitude, then combined, then the duplicate 'month' columns are removed.
    cycle_list_with_each_station = [x for _, x
                                    in sorted(zip(list(df_station_metadata['lat']), cycle_list_with_each_station))]
    df_cycles_for_all_stations = pd.concat(cycle_list_with_each_station, axis=1, sort=False)
    df_cycles_for_all_stations = df_cycles_for_all_stations.loc[:, ~df_cycles_for_all_stations.columns.duplicated()]
    df_station_metadata.sort_values(by='lat', ascending=True, inplace=True)  # sort the metadata after using it for sorting the cycle list

    # --- Plot the seasonal cycles for all stations
    xdata = df_cycles_for_all_stations['month']
    ydata = df_cycles_for_all_stations.loc[:, (df_cycles_for_all_stations.columns != 'month')]
    station_names = ydata.columns.values
    #
    if compare_against_model:
        plot_comparison_against_model(ref_dt, ref_vals, f'obs [{opts.station_code}]',
                                      mdl_dt, mdl_vals, f'model [{opts.model_name}]',
                                      savepath=opts.figure_savepath)
    else:
        plot_lines_for_all_station_cycles(xdata, ydata, savepath=opts.figure_savepath)
        plot_heatmap_of_all_stations(xdata, ydata, latitudes=list(df_station_metadata['lat']), savepath=opts.figure_savepath)

    # --- Make a supplemental figure for filter components ---
    #
    # fig, axs = plt.subplots(2, 2, sharex='all', sharey='row', figsize=(14, 7))
    # ax_iterator = np.ndenumerate(axs)
    #
    # _, ax = next(ax_iterator)
    # ax.plot(x, detrend, label='detrend', alpha=0.2, marker='.')
    # aesthetic_grid_no_spines(ax)
    # ax.legend()
    # #
    # _, ax = next(ax_iterator)
    # ax.plot(x, resid_from_func, label='residuals from the function', alpha=0.2, marker='.')
    # ax.plot(x, resid_from_smooth, label='residuals about the smoothed line', alpha=0.2, marker='.')
    # aesthetic_grid_no_spines(ax)
    # ax.legend()
    # #
    # _, ax = next(ax_iterator)
    # ax.plot(x1, y9, label='equally spaced interpolated data with function removed', alpha=0.2, marker='.')
    # aesthetic_grid_no_spines(ax)
    # ax.legend()
    # #
    # _, ax = next(ax_iterator)
    # ax.plot(x0, resid_smooth, label='smoothed residuals', alpha=0.2, marker='.', color='gray')
    # ax.plot(x0, resid_trend, label='trend of residuals', alpha=0.2, marker='.')
    # aesthetic_grid_no_spines(ax)
    # ax.legend()
    # #
    # plt.tight_layout()
    # #
    # if opts.figure_savepath:
    #     mysavefig(fig=fig, plot_save_name=opts.figure_savepath + 'supplement2.png')

    if compare_against_model:
        returnval = df_cycles_for_all_stations, cycle_list_with_each_station, df_station_metadata, mdl_dt, mdl_vals
    else:
        returnval = df_cycles_for_all_stations, cycle_list_with_each_station, df_station_metadata
    return returnval


def update_for_skipped_station(msg, station_name, station_count, counter_dict):
    """Print a message and reduce the total station count by one."""
    _logger.info('  %s', msg)
    _logger.info('  skipping station: %s', station_name)
    counter_dict['skipped'] += 1
    station_count[0] -= 1


def sort_lists_by(lists, key_list=0, desc=False):
    """
    From https://stackoverflow.com/a/15611016

    Parameters
    ----------
    lists
    key_list
    desc

    Returns
    -------
    sorted lists
    """
    return zip(*sorted(zip(*lists), reverse=desc,
                 key=lambda x: x[key_list]))


def make_cycle(x0, smooth_cycle) -> tuple:
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

def plot_comparison_against_model(ref_xdata: pd.DataFrame,
                                  ref_ydata: pd.DataFrame,
                                  ref_label: str,
                                  mdl_xdata: pd.DataFrame,
                                  mdl_ydata: pd.DataFrame,
                                  mdl_label: str,
                                  savepath=None) -> None:
    # --- Plot the seasonal cycle
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(ref_xdata, ref_ydata, label=ref_label, marker='o', color='k')
    ax.plot(mdl_xdata, mdl_ydata, label=mdl_label, marker='o', color='r')
    #
    plt.title('annual climatology')
    ax.set_ylabel("$CO_2$ (ppm)")
    #
    # Specify the xaxis tick labels format -- %b gives us Jan, Feb...
    month_fmt = mdates.DateFormatter('%b')
    # ax.xaxis.set_major_formatter(month_fmt)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda z, pos=None: month_fmt(z)[0]))
    #
    aesthetic_grid_no_spines(ax)
    #
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    #
    plt.tight_layout()
    if savepath:
        mysavefig(fig=fig, plot_save_name=savepath + 'supplement_compare_against_model_lines.png')

def plot_lines_for_all_station_cycles(xdata: pd.DataFrame,
                                      ydata: pd.DataFrame,
                                      savepath=None) -> None:
    # --- Plot the seasonal cycle for all stations
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(xdata, ydata, '-o')
    #
    aesthetic_grid_no_spines(ax)
    #
    plt.legend(ydata.columns.values, loc='upper left')
    #
    # Specify the xaxis tick labels format -- %b gives us Jan, Feb...
    month_fmt = mdates.DateFormatter('%b')
    # ax.xaxis.set_major_formatter(month_fmt)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda z, pos=None: month_fmt(z)[0]))
    #
    plt.tight_layout()
    if savepath:
        mysavefig(fig=fig, plot_save_name=savepath + 'supplement_allstations_lines.png')


def plot_heatmap_of_all_stations(xdata: pd.DataFrame,
                                 ydata: pd.DataFrame,
                                 latitudes: list = None,
                                 savepath=None) -> None:
    mindate = mdates.date2num(xdata.tolist()[0])
    maxdate = mdates.date2num(xdata.tolist()[-1])

    num_stations = ydata.shape[1]
    station_labels = list(ydata.columns.values)

    # --- Plot the seasonal cycle for all stations,
    #   and flip the ydata because pyplot.imshow will plot the last row on the bottom
    fig, ax = plt.subplots(1, 1, figsize=(10, num_stations*0.8))
    im = ax.imshow(ydata.transpose().iloc[::-1], cmap='viridis', interpolation='nearest',
                   aspect='auto',
                   extent=(mindate, maxdate, -0.5, num_stations - 0.5))
    #
    # y_label_list = station_names
    ax.set_yticks(range(num_stations))
    ax.set_yticklabels(station_labels)
    #
    # Add secondary y-axis on the right side to show station latitudes
    if latitudes:
        ax2 = ax.twinx()
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(num_stations))
        ax2.set_yticklabels([numstr(x, decimalpoints=2) for x in latitudes])
    #
    cbar = fig.colorbar(im, orientation="horizontal", pad=0.2)
    cbar.ax.set_xlabel('$CO_2$ (ppm)')
    #
    # Specify the xaxis tick labels format -- %b gives us Jan, Feb...
    month_fmt = mdates.DateFormatter('%b')
    # ax.xaxis.set_major_formatter(month_fmt)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda z, pos=None: month_fmt(z)[0]))
    #
    if savepath:
        mysavefig(fig=fig, plot_save_name=savepath + 'supplement_allstations_heatmap.png')


def plot_filter_components(filter_object, original_x, original_y,
                           figure_title='', savepath=None) -> None:
    # --- Make the figure ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(filter_object.xinterp, filter_object.getFunctionValue(filter_object.xinterp), label='Function values',
            color=[x / 255 for x in [255, 127, 14]], alpha=1, linewidth=2.5, )
    ax.plot(filter_object.xinterp, filter_object.getPolyValue(filter_object.xinterp), label='Poly values',
            color=[x / 255 for x in [31, 119, 180]], alpha=1, linewidth=2.5, )
    ax.plot(filter_object.xinterp, filter_object.getTrendValue(filter_object.xinterp), label='Trend values',
            color=[x / 255 for x in [44, 160, 44]], alpha=1, linewidth=2.5, )
    # ax.plot(x0, y3, label='Smooth values',
    #        alpha=1, linewidth=2.5, )
    ax.plot(original_x, original_y, label='original',
            marker='.', linestyle='none', color='gray', zorder=-10, alpha=0.2)
    ax.set_ylabel("$CO_2$ (ppm)")
    ax.set_xlabel("year")
    #
    aesthetic_grid_no_spines(ax)
    #
    plt.title(figure_title)
    #
    plt.legend()
    #
    plt.tight_layout()
    #
    if savepath:
        mysavefig(fig=fig, plot_save_name=savepath + 'supplement1_mdl.png')

def t2dt(atime):
    """
    Convert atime (a float) to DT.datetime
    This is the inverse of dt2t.
    assert dt2t(t2dt(atime)) == atime
    """
    year = int(atime)
    remainder = atime - year
    boy = datetime.datetime(year, 1, 1)
    eoy = datetime.datetime(year + 1, 1, 1)
    seconds = remainder * (eoy - boy).total_seconds()
    return boy + timedelta(seconds=seconds)


def dt2t(year, month, day, h=0, m=0, s=0) :
    """convert a DT.datetime to a float"""
    year_seconds = (datetime.datetime(year,12,31,23,59,59,999999)-datetime.datetime(year,1,1,0,0,0)).total_seconds()
    second_of_year = (datetime.datetime(year,month,day,h,m,s) - datetime.datetime(year,1,1,0,0,0)).total_seconds()
    return year + second_of_year / year_seconds


def add_seasonal_cycle_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """Add recipe arguments to a parser object

    Parameters
    ----------
    parser
    """
    add_shared_arguments_for_recipes(parser)
    parser.add_argument('--model_name', default='CMIP.NOAA-GFDL.GFDL-ESM4.esm-hist.Amon.gr1',
                        type=cmip_collection_module.model_substring, choices=cmip_collection_module.model_choices)
    parser.add_argument('--station_code', default='mlo',
                        type=str, choices=obspack_surface_collection_module.station_dict.keys())
    parser.add_argument('--difference', action='store_true')
    parser.add_argument('--run_all_stations', action='store_true')
    parser.add_argument('--run_three_stations', action='store_true')
    parser.add_argument('--plot_filter_components', action='store_true')
    parser.add_argument('--globalmean', action='store_true')
    parser.add_argument('--use_mlo_for_detrending', action='store_true')
    parser.add_argument('--station_list', nargs='*')
