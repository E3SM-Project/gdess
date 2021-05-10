""" This produces a plot of multidecadal trends of atmospheric CO2
This function parses:
 - observational data from Globalview+ surface stations
 - model output from CMIP6
================================================================================
"""
import argparse
import datetime
from datetime import timedelta
import tempfile
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
from co2_diag.operations.geographic import get_closest_mdl_cell_dict
from co2_diag.operations.time import ensure_dataset_datetime64, year_to_datetime64
from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig, limits_with_zero
from co2_diag.recipes.utils import valid_year_string, options_to_args

from ccgcrv.ccgcrv import ccgcrv

import logging
_logger = logging.getLogger(__name__)


def seasonal_cycles(options: dict,
                    verbose: Union[bool, str] = False,
                    ):
    """Execute a series of preprocessing steps and generate a diagnostic result.

    Relevant co2_diag collections are instantiated and processed.

    Parameters
    ----------
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
    verbose: Union[bool, str]
        can be either True, False, or a string for level such as "INFO, DEBUG, etc."

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

    if opts.use_mlo_for_detrending:
        ds_mlo_ref = obs_collection.stepA_original_datasets['mlo']

    df_surface_station = ds_obs['co2'].to_dataframe().reset_index()

    # --- Fit a curve to the surface station data ---
    n_poly_terms = 3
    n_harm_terms = 4
    today_str = datetime.datetime.today().strftime('%Y-%m-%d')
    output_filepath = f'curve_fit_results_{n_poly_terms}poly_{n_harm_terms}harm_{today_str}.txt'

    # Write dataframe to temporary file, read it, and fit the curve
    with tempfile.NamedTemporaryFile(suffix='.txt', prefix=('test_mlo'),
                                     delete=False, mode='w+') as temp:
        df_surface_station.loc[:, ['time_decimal', 'co2']].to_csv(temp.name, sep=' ', index=False, header=False)

        options = {'npoly': n_poly_terms,
                   'nharm': n_harm_terms,
                   'file': output_filepath,
                   'short': 400,
                   'equal': '',
                   'showheader': '',
                   'func': '',
                   'poly': '',
                   'trend': '',
                   'res': ''}

        if _logger.level < 20:
            # log level is lower than "INFO" (20), e.g. "VERBOSE" (15) or DEBUG (10)
            options['stats'] = ''  # print stats output
            options['amp'] = ''  # print amplitudes

        filt = ccgcrv(options, temp.name)

    # Show the first few rows
    df_ccgcrv_output = pd.read_csv(output_filepath, sep='\s+')
    df_ccgcrv_output['datetime'] = df_ccgcrv_output.date.map(t2dt)
    _logger.debug("filter output head: %s", df_ccgcrv_output.head(2))

    # Original data points
    x = df_surface_station['time_decimal'].values
    y = df_surface_station['co2'].values

    # --- Extract the relevant filtered components ---
    x0 = filt.xinterp
    y1 = filt.getFunctionValue(x0)
    y2 = filt.getPolyValue(x0)
    y3 = filt.getSmoothValue(x0)
    y4 = filt.getTrendValue(x0)

    # Seasonal Cycle
    trend = filt.getTrendValue(x)
    detrend = y - trend
    harmonics = filt.getHarmonicValue(x0)
    smooth_cycle = harmonics + filt.smooth - filt.trend

    # residuals from the function
    resid_from_func = filt.resid

    # smoothed residuals
    resid_smooth = filt.smooth

    # trend of residuals
    resid_trend = filt.trend

    # residuals about the smoothedline
    resid_from_smooth = filt.yp - filt.getSmoothValue(x)

    # equally spaced interpolated data with function removed
    x1 = filt.xinterp
    y9 = filt.yinterp

    # --- Make the figure ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(df_surface_station['time'], df_surface_station['co2'], label='MLO obs',
            marker='.', linestyle='none', color='gray', zorder=-10, alpha=0.2)
    ax.plot(df_ccgcrv_output['datetime'], df_ccgcrv_output['function'], label='Curve fit function',
            alpha=1, linewidth=2.5, color=[x / 255 for x in [255, 127, 14]])
    ax.plot(df_ccgcrv_output['datetime'], df_ccgcrv_output['polynomial'], label='Curve fit polynomial',
            alpha=1, linewidth=2.5, color=[x / 255 for x in [31, 119, 180]])
    ax.plot(df_ccgcrv_output['datetime'], df_ccgcrv_output['trend'], label='Curve fit trend',
            alpha=1, linewidth=2.5, color=[x / 255 for x in [44, 160, 44]])
    ax.set_ylabel("$CO_2$ (ppm)")
    ax.set_xlabel("year")
    #
    aesthetic_grid_no_spines(ax)
    #
    plt.legend()
    #
    plt.tight_layout()
    #
    if opts.figure_savepath:
        mysavefig(fig=fig, plot_save_name=opts.figure_savepath + 'supplement1.png')

    # --- Make a supplemental figure for filter components ---

    fig, axs = plt.subplots(2, 2, sharex='all', sharey='row', figsize=(14, 7))
    ax_iterator = np.ndenumerate(axs)

    _, ax = next(ax_iterator)
    ax.plot(x, detrend, label='detrend', alpha=0.2, marker='.')
    aesthetic_grid_no_spines(ax)
    ax.legend()
    #
    _, ax = next(ax_iterator)
    ax.plot(x, resid_from_func, label='residuals from the function', alpha=0.2, marker='.')
    ax.plot(x, resid_from_smooth, label='residuals about the smoothed line', alpha=0.2, marker='.')
    aesthetic_grid_no_spines(ax)
    ax.legend()
    #
    _, ax = next(ax_iterator)
    ax.plot(x1, y9, label='equally spaced interpolated data with function removed', alpha=0.2, marker='.')
    aesthetic_grid_no_spines(ax)
    ax.legend()
    #
    _, ax = next(ax_iterator)
    ax.plot(x0, resid_smooth, label='smoothed residuals', alpha=0.2, marker='.', color='gray')
    ax.plot(x0, resid_trend, label='trend of residuals', alpha=0.2, marker='.')
    aesthetic_grid_no_spines(ax)
    ax.legend()
    #
    plt.tight_layout()
    #
    if opts.figure_savepath:
        mysavefig(fig=fig, plot_save_name=opts.figure_savepath + 'supplement2.png')

    # --- Do a seasonal climatology ---

    # Convert dates to datetime objects, and make a dataframe with a month column for grouping purposes.
    df_seasonalcycle = pd.DataFrame.from_dict({'datetime': [t2dt(i) for i in x0],
                                               'co2': smooth_cycle})
    df_seasonalcycle['month'] = df_seasonalcycle['datetime'].dt.month

    # Bin by month, and add a column that represents months in datetime format for plotting purposes.
    df_monthly = df_seasonalcycle.groupby('month').mean().reset_index()
    df_monthly['month_datetime'] = pd.to_datetime(df_monthly['month'], format='%m')

    # --- Plot the seasonal cycle
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(df_monthly['month_datetime'], df_monthly['co2'],
            label='seasonal climatology', marker='o')

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
    plt.legend()
    #
    plt.tight_layout()
    #
    if opts.figure_savepath:
        mysavefig(fig=fig, plot_save_name=opts.figure_savepath)


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


def _parse_options(params: dict):
    _logger.debug("Parsing diagnostic parameters...")

    param_argstr = options_to_args(params)
    _logger.debug('Parameter argument string == %s', param_argstr)

    parser = argparse.ArgumentParser(description='Process surface observing station and CMIP data and compare. ')
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
    parser.add_argument('--use_mlo_for_detrending', action='store_true')
    args = parser.parse_args(param_argstr)

    # Convert times to numpy.datetime64
    args.start_datetime = year_to_datetime64(args.start_yr)
    args.end_datetime = year_to_datetime64(args.end_yr)

    _logger.debug("Parsing is done.")
    return args