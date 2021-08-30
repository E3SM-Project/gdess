""" This produces plots of seasonal cycles of atmospheric CO2
This function parses:
 - observational data from Globalview+ surface stations
 - model output from CMIP6
================================================================================
"""
from co2_diag import set_verbose, load_stations_dict
from co2_diag.formatters import append_before_extension, numstr
from co2_diag.data_source.cmip import Collection as cmipCollection, matched_model_and_experiment, model_choices
from co2_diag.operations.Confrontation import make_comparable, apply_time_bounds
from co2_diag.operations.time import t2dt
from co2_diag.recipes.utils import add_shared_arguments_for_recipes, parse_recipe_options, benchmark_recipe
from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig
import co2_diag.data_source.obspack.gvplus_surface as obspack_surface_collection_module

from ccgcrv.ccg_filter import ccgFilter
from ccgcrv.ccg_dates import decimalDateFromDatetime

import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib import ticker
from typing import Union
from datetime import datetime
import argparse, logging, csv

_logger = logging.getLogger(__name__)

stations_dict = load_stations_dict()

@benchmark_recipe
def seasonal_cycles(options: Union[dict, argparse.Namespace],
                    verbose: Union[bool, str] = False,
                    ) -> tuple:
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
            cmip_load_method (str):
                either 'pangeo' (which uses a stored url),
                or 'local' (which uses the path defined in config file)
            start_yr (str): '1960' is default
            end_yr (str): '2015' is default
            latitude_bin_size (numeric): None is default
            figure_savepath (str): None is default
            difference (str): None is default
            globalmean (str):
                either 'station', which requires specifying the <station_code> parameter,
                or 'global', which will calculate a global mean
    verbose: Union[bool, str]
        can be either True, False, or a string for level such as "INFO, DEBUG, etc."

    Returns
    -------
    A tuple:
        A DataFrame containing the data that were plotted.
        A list of the data for each station
        A DataFrame containing the metadata for each station
        (and if a comparison with a model was made, then the datetimes and values are also part of the returned tuple)
    """
    set_verbose(_logger, verbose)
    if verbose:
        ProgressBar().register()
    _logger.debug("Parsing diagnostic parameters...")
    opts = parse_recipe_options(options, add_seasonal_cycle_args_to_parser)

    # The list of stations to analyze is populated.
    if opts.run_all_stations:
        stations_to_analyze = obspack_surface_collection_module.station_dict.keys()
    elif opts.station_list:
        stations_to_analyze = opts.station_list
    else:
        stations_to_analyze = [opts.station_code]

    # --- Load CMIP model output ---
    # We will only compare against CMIP model outputs if a model_name is supplied
    if compare_against_model := bool(opts.model_name):
        _logger.info('*Processing CMIP model output*')
        new_self, _ = cmipCollection._recipe_base(datastore='cmip6', verbose=verbose,
                                                  load_method=opts.cmip_load_method, skip_selections=True)
        ds_mdl = new_self.stepB_preprocessed_datasets[opts.model_name]
        ds_mdl = ds_mdl.assign_coords(time_decimal=('time', [decimalDateFromDatetime(x)
                                                             for x in pd.DatetimeIndex(ds_mdl['time'].values)]))
    else:
        ds_mdl = None

    # --- Observation data is processed for each station location. ---
    _logger.info('*Processing Observations*')
    counter = {'current': 1, 'skipped': 0}
    processed_station_metadata = dict(lat=[], lon=[], code=[], fullname=[])
    cycles_of_each_station = dict(ref=[], mdl=[])
    num_stations = [len(stations_to_analyze)]
    for station in stations_to_analyze:
        _logger.info("Station %s of %s: %s", counter['current'], num_stations[0], station)
        obs_collection = obspack_surface_collection_module.Collection(verbose=verbose)
        obs_collection.preprocess(datadir=opts.ref_data, station_name=station)
        ds_obs = obs_collection.stepA_original_datasets[station]
        _logger.info('  %s', obs_collection.station_dict.get(station))

        # if opts.use_mlo_for_detrending:
        #     ds_mlo_ref = obs_collection.stepA_original_datasets['mlo']

        # Apply time bounds, and get the relevant model output.
        try:
            if compare_against_model:
                da_obs, da_mdl = make_comparable(ds_obs, ds_mdl,
                                                 time_limits=(np.datetime64(opts.start_yr), np.datetime64(opts.end_yr)),
                                                 latlon=(ds_obs['latitude'].values[0], ds_obs['longitude'].values[0]),
                                                 altitude=ds_obs['altitude'].values[0], altitude_method='lowest',
                                                 global_mean=False, verbose=verbose)
            else:
                da_obs, _, _ = apply_time_bounds(ds_obs, time_limits=(np.datetime64(opts.start_yr),
                                                                      np.datetime64(opts.end_yr)))
        except (RuntimeError, AssertionError) as re:
            update_for_skipped_station(re, station, num_stations, counter)
            continue
        #
        # Check that there is at least one year's worth of data for this station.
        if (da_obs.time.values.max().astype('datetime64[M]') - da_obs.time.values.min().astype('datetime64[M]')) < 12:
            msg = '  insufficient number of months of data for station <%s>' % station
            update_for_skipped_station(msg, station, num_stations, counter)
            continue

        # --- Curve fitting ---
        #   (i) Globalview+ data
        filt_ref = ccgFilter(xp=da_obs['time_decimal'].values, yp=da_obs['co2'].values,
                             numpolyterms=3, numharmonics=4, timezero=int(da_obs['time_decimal'].values[0]))
        #   (ii) CMIP data
        if compare_against_model:
            try:
                filt_mdl = ccgFilter(xp=da_mdl['time_decimal'].values, yp=da_mdl.values,
                                     numpolyterms=3, numharmonics=4, timezero=int(da_mdl['time_decimal'].values[0]))
            except TypeError as te:
                _logger.info('--- Curve filtering error ---')
                update_for_skipped_station(te, station, num_stations, counter)
                continue

        # Optional plotting of components of the filtering process
        if opts.plot_filter_components:
            plot_filter_components(filt_ref,
                                   original_x=ds_obs['time_decimal'].values, #df_surface_station['time_decimal'].values,
                                   original_y=ds_obs['co2'].values,  #df_surface_station['co2'].values,
                                   figure_title=f'obs, station {station}',
                                   savepath=append_before_extension(opts.figure_savepath, '_supplement1ref_' + station))
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
        cycles_of_each_station['ref'].append(pd.DataFrame.from_dict({"month": ref_dt, f"{station}": ref_vals}))
        #   (ii) CMIP data
        if compare_against_model:
            mdl_dt, mdl_vals = make_cycle(x0=filt_mdl.xinterp,
                                          smooth_cycle=filt_mdl.getHarmonicValue(
                                              filt_mdl.xinterp) + filt_mdl.smooth - filt_mdl.trend)
            cycles_of_each_station['mdl'].append(pd.DataFrame.from_dict({"month": mdl_dt, f"{station}": mdl_vals}))

        # Gather together the station's metadata at the loop end, when we're sure that this station has been processed.
        processed_station_metadata['lon'].append(obs_collection.station_dict[station]['lon'])
        processed_station_metadata['lat'].append(obs_collection.station_dict[station]['lat'])
        processed_station_metadata['fullname'].append(obs_collection.station_dict[station]['name'])
        processed_station_metadata['code'].append(station)
        counter['current'] += 1
        # END of station loop

    _logger.info("Done -- %s stations fully processed. %s stations skipped.",
                 len(cycles_of_each_station['ref']), counter['skipped'])

    # Dataframes for each location are combined so we have one 'month' column, and a single column for each station.
    # First, dataframes are sorted by latitude, then combined, then the duplicate 'month' columns are removed.
    df_station_metadata = pd.DataFrame.from_dict(processed_station_metadata)
    df_all_cycles = dict(ref=None, mdl=None)
    #   (i) Globalview+ data
    cycles_of_each_station['ref'] = [x for _, x
                                     in sorted(zip(list(df_station_metadata['lat']), cycles_of_each_station['ref']))]
    df_all_cycles['ref'] = pd.concat(cycles_of_each_station['ref'], axis=1, sort=False)
    df_all_cycles['ref'] = df_all_cycles['ref'].loc[:, ~df_all_cycles['ref'].columns.duplicated()]
    #   (ii) CMIP data
    if compare_against_model:
        cycles_of_each_station['mdl'] = [x for _, x
                                         in sorted(zip(list(df_station_metadata['lat']), cycles_of_each_station['mdl']))]
        df_all_cycles['mdl'] = pd.concat(cycles_of_each_station['mdl'], axis=1, sort=False)
        df_all_cycles['mdl'] = df_all_cycles['mdl'].loc[:, ~df_all_cycles['mdl'].columns.duplicated()]
    #
    # Sort the metadata after using it for sorting the cycle list(s)
    df_station_metadata.sort_values(by='lat', ascending=True, inplace=True)

    # --- Optional binning by latitude ---
    if opts.latitude_bin_size:
        # we won't use additional latitude labels for the heatmap, because the left side will be latitude bins
        heatmap_rightside_labels = None

        # We determine bins to which each station is assigned.
        def to_bin(x): return np.floor(x / opts.latitude_bin_size) * opts.latitude_bin_size
        df_station_metadata["latbin"] = df_station_metadata['lat'].map(to_bin)
        df_station_metadata["lonbin"] = df_station_metadata['lon'].map(to_bin)
        #
        df_all_cycles['ref'] = calc_binned_means(df_all_cycles['ref'], df_station_metadata)
        if compare_against_model:
            df_all_cycles['mdl'] = calc_binned_means(df_all_cycles['mdl'], df_station_metadata)
    else:
        heatmap_rightside_labels = [numstr(x, decimalpoints=2) for x in df_station_metadata['lat']]

    # Write output data to csv
    filename = append_before_extension(opts.figure_savepath + '.csv',
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

    # --- MAKE PLOTS ---
    # --- Plot lineplot comparing model and observations
    # if compare_against_model:
    #     pass
    #     plot_comparison_against_model(ref_dt, ref_vals, f'obs [{opts.station_code}]',
    #                                   mdl_dt, mdl_vals, f'model [{opts.model_name}]',
    #                                   savepath=opts.figure_savepath)
    # else:
    # --- Plot the seasonal cycles at all station locations
    xdata = df_all_cycles['ref']['month']
    #   (i) Globalview+ data
    ydata_gv = df_all_cycles['ref'].loc[:, (df_all_cycles['ref'].columns != 'month')]
    plot_lines_for_all_station_cycles(xdata, ydata_gv.iloc[:, ::-1], figure_title="GV+",
                                      savepath=append_before_extension(opts.figure_savepath, '_obs_lineplot'))
    plot_heatmap_of_all_stations(xdata, ydata_gv, rightside_labels=heatmap_rightside_labels, figure_title="obs",
                                 savepath=append_before_extension(opts.figure_savepath, '_obs_heatmap'))

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

    if compare_against_model:
        if not xdata.equals(df_all_cycles['mdl']['month']):
            raise ValueError('Unexpected discrepancy, xdata for reference observations does not equal xdata for models')

        #   (ii) CMIP data
        ydata_cmip = df_all_cycles['mdl'].loc[:, (df_all_cycles['mdl'].columns != 'month')]
        plot_lines_for_all_station_cycles(xdata, ydata_cmip.iloc[:, ::-1], figure_title="CMIP",
                                          savepath=append_before_extension(opts.figure_savepath, '_mdl_lineplot'))
        plot_heatmap_of_all_stations(xdata, ydata_cmip, rightside_labels=heatmap_rightside_labels, figure_title="mdl",
                                     savepath=append_before_extension(opts.figure_savepath, '_mdl_heatmap'))

        #   (iii) Model - obs difference
        ydiff = ydata_cmip - ydata_gv
        plot_lines_for_all_station_cycles(xdata, ydiff.iloc[:, ::-1], figure_title="Difference",
                                          savepath=append_before_extension(opts.figure_savepath, '_diff_lineplot'))
        plot_heatmap_of_all_stations(xdata, ydiff, rightside_labels=heatmap_rightside_labels,
                                     figure_title=f"model - obs",
                                     savepath=append_before_extension(opts.figure_savepath, '_diff_heatmap'))

        # Write output data for this instance
        for column in ydata_cmip:
            row_dict = {
                'station': column,
                'source': 'cmip',
                'max': ydata_cmip[column].max(),
                'min': ydata_cmip[column].min(),
                'mean': ydata_cmip[column].mean(),
                'median': ydata_cmip[column].median(),
                'std': ydata_cmip[column].std(),
                'rmse': mean_squared_error(ydata_gv[column], ydata_cmip[column], squared=False)
            }
            writer.writerow(row_dict)

    fileptr.flush()

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
    #     mysavefig(fig=fig, plot_save_name=append_before_extension(opts.figure_savepath, 'supplement2'))

    return df_all_cycles, cycles_of_each_station, df_station_metadata


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
    lgd = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    #
    plt.tight_layout()
    if savepath:
        mysavefig(fig=fig, plot_save_name=append_before_extension(savepath, 'supplement_compare_against_model_lines'),
                  bbox_inches='tight', bbox_extra_artists=(lgd, ))


def plot_lines_for_all_station_cycles(xdata: pd.DataFrame,
                                      ydata: pd.DataFrame,
                                      figure_title: str = '',
                                      savepath=None) -> None:
    # --- Plot the seasonal cycle for all stations
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(xdata, ydata, '-o')
    #
    aesthetic_grid_no_spines(ax)
    #
    lgd = plt.legend(ydata.columns.values, loc='upper left')
    #
    ax.set_ylabel('$CO_2$ (ppm)')
    if figure_title:
        ax.set_title(figure_title)
    #
    # Specify the xaxis tick labels format -- %b gives us Jan, Feb...
    month_fmt = mdates.DateFormatter('%b')
    # ax.xaxis.set_major_formatter(month_fmt)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda z, pos=None: month_fmt(z)[0]))
    #
    plt.tight_layout()
    if savepath:
        mysavefig(fig=fig, plot_save_name=savepath, bbox_inches='tight', bbox_extra_artists=(lgd, ))


def plot_heatmap_of_all_stations(xdata: pd.DataFrame,
                                 ydata: pd.DataFrame,
                                 rightside_labels: list = None,
                                 figure_title: str = '',
                                 savepath=None) -> None:
    mindate = mdates.date2num(xdata.tolist()[0])
    maxdate = mdates.date2num(xdata.tolist()[-1])

    num_stations = ydata.shape[1]
    station_labels = list(ydata.columns.values)

    # --- Plot the seasonal cycle for all stations,
    #   and flip the ydata because pyplot.imshow will plot the last row on the bottom
    totalfigheight = (num_stations*0.8) + (num_stations*0.8)*0.15  # Add 0.15 for the colorbar
    fig, ax = plt.subplots(1, 1, figsize=(6, totalfigheight))
    im = ax.imshow(ydata.transpose().iloc[::-1],
                   norm=mcolors.TwoSlopeNorm(vcenter=0.), cmap='RdBu_r', interpolation='nearest',
                   aspect='auto', extent=(mindate, maxdate, -0.5, num_stations - 0.5))
    #
    # y_label_list = station_names
    ax.set_yticks(range(num_stations))
    ax.set_yticklabels(station_labels)
    #
    # Add secondary y-axis on the right side to show station latitudes
    if rightside_labels:
        ax2 = ax.twinx()
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(num_stations))
        ax2.set_yticklabels(rightside_labels)
    #
    cbar = fig.colorbar(im, orientation="horizontal", pad=max([0.5, (num_stations - 0.5)*0.008]))
    cbar.ax.set_xlabel('$CO_2$ (ppm)')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='vertical')
    #
    # Specify the xaxis tick labels format -- %b gives us Jan, Feb...
    month_fmt = mdates.DateFormatter('%b')
    # ax.xaxis.set_major_formatter(month_fmt)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda z, pos=None: month_fmt(z)[0]))
    #
    if figure_title:
        ax.set_title(figure_title)
    #
    if savepath:
        mysavefig(fig=fig, plot_save_name=savepath, bbox_inches='tight')


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
    lgd = plt.legend()
    #
    plt.tight_layout()
    #
    if savepath:
        mysavefig(fig=fig, plot_save_name=append_before_extension(savepath, '_filter_components'),
                  bbox_inches='tight', bbox_extra_artists=(lgd, ))


def add_seasonal_cycle_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """Add recipe arguments to a parser object

    Parameters
    ----------
    parser
    """
    add_shared_arguments_for_recipes(parser)
    parser.add_argument('--model_name', default='',
                        type=matched_model_and_experiment, choices=model_choices)
    parser.add_argument('--cmip_load_method', default='pangeo',
                        type=str, choices=['pangeo', 'local'])
    parser.add_argument('--station_code', default='mlo',
                        type=str, choices=obspack_surface_collection_module.station_dict.keys())
    parser.add_argument('--difference', action='store_true')
    parser.add_argument('--latitude_bin_size', default=None, type=float)
    parser.add_argument('--plot_filter_components', action='store_true')
    parser.add_argument('--globalmean', action='store_true')
    parser.add_argument('--use_mlo_for_detrending', action='store_true')
    parser.add_argument('--run_all_stations', action='store_true')
    parser.add_argument('--station_list', nargs='*')
