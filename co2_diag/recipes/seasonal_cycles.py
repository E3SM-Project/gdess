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
from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig, limits_with_zero
from co2_diag.recipes.utils import add_shared_arguments_for_recipes, parse_recipe_options

from co2_diag.operations.Confrontation import make_comparable

from ccgcrv.ccg_filter import ccgFilter
from ccgcrv.ccg_dates import datetimeFromDecimalDate, calendarDate, decimalDateFromDatetime

import logging
_logger = logging.getLogger(__name__)


def seasonal_cycles(options: Union[dict, argparse.Namespace],
                    verbose: Union[bool, str] = False,
                    ):
    """Execute a series of preprocessing steps and generate a diagnostic result.

    Relevant co2_diag collections are instantiated and processed.

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

    # --- Surface observations ---
    _logger.info('*Processing Observations*')
    obs_collection = obspack_surface_collection_module.Collection(verbose=verbose)
    obs_collection.preprocess(datadir=opts.ref_data, station_name=opts.station_code)
    ds_obs = obs_collection.stepA_original_datasets[opts.station_code]
    _logger.info('%s', obs_collection.station_dict[opts.station_code])

    if opts.use_mlo_for_detrending:
        ds_mlo_ref = obs_collection.stepA_original_datasets['mlo']

    df_surface_station = ds_obs['co2'].to_dataframe().reset_index()

    # --- CMIP model output ---
    _logger.info('*Processing CMIP model output*')
    cmip_collection = cmip_collection_module.Collection(verbose=verbose)
    new_self, loaded_from_file = cmip_collection._recipe_base(datastore='cmip6', verbose=verbose,
                                                              from_file=None, skip_selections=True)
    ds_mdl = new_self.stepB_preprocessed_datasets[opts.model_name]

    da_obs, da_mdl = make_comparable(ds_obs, ds_mdl,
                                     time_limits=(np.datetime64(opts.start_yr),
                                                  np.datetime64(opts.end_yr)),
                                     latlon=(ds_obs['latitude'].values[0], ds_obs['longitude'].values[0]),
                                     global_mean=False)

    # Surface station curve fitting
    xp = da_obs['time_decimal'].values
    yp = da_obs['co2'].values
    filt_ref = ccgFilter(xp=xp, yp=yp, numpolyterms=3, numharmonics=4, timezero=int(xp[0]))

    # CMIP data curve fitting

    da_mdl = da_mdl.assign_coords(time_decimal=('time',
                                                [decimalDateFromDatetime(x) for x in
                                                 pd.DatetimeIndex(da_mdl['time'].values)]))
    xp = da_mdl['time_decimal'].values
    yp = da_mdl.values
    filt_mdl = ccgFilter(xp=xp, yp=yp, numpolyterms=3, numharmonics=4, timezero=int(xp[0]))

    # Original data points
    x = df_surface_station['time_decimal'].values
    y = df_surface_station['co2'].values

    # --- Make the figure ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(filt_ref.xinterp, filt_ref.getFunctionValue(filt_ref.xinterp), label='Function values',
            color=[x / 255 for x in [255, 127, 14]], alpha=1, linewidth=2.5, )
    ax.plot(filt_ref.xinterp, filt_ref.getPolyValue(filt_ref.xinterp), label='Poly values',
            color=[x / 255 for x in [31, 119, 180]], alpha=1, linewidth=2.5, )
    ax.plot(filt_ref.xinterp, filt_ref.getTrendValue(filt_ref.xinterp), label='Trend values',
            color=[x / 255 for x in [44, 160, 44]], alpha=1, linewidth=2.5, )
    # ax.plot(x0, y3, label='Smooth values',
    #        alpha=1, linewidth=2.5, )
    ax.plot(df_surface_station['time_decimal'].values, df_surface_station['co2'].values,
            label='original',
            marker='.', linestyle='none', color='gray', zorder=-10, alpha=0.2)
    ax.set_ylabel("$CO_2$ (ppm)")
    ax.set_xlabel("year")
    #
    plt.title('obs')
    #
    aesthetic_grid_no_spines(ax)
    #
    plt.legend()
    #
    plt.tight_layout()
    #
    if opts.figure_savepath:
        mysavefig(fig=fig, plot_save_name=opts.figure_savepath + 'supplement1_ref.png')

    # --- Make the figure ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(filt_mdl.xinterp, filt_mdl.getFunctionValue(filt_mdl.xinterp), label='Function values',
            color=[x / 255 for x in [255, 127, 14]], alpha=1, linewidth=2.5, )
    ax.plot(filt_mdl.xinterp, filt_mdl.getPolyValue(filt_mdl.xinterp), label='Poly values',
            color=[x / 255 for x in [31, 119, 180]], alpha=1, linewidth=2.5, )
    ax.plot(filt_mdl.xinterp, filt_mdl.getTrendValue(filt_mdl.xinterp), label='Trend values',
            color=[x / 255 for x in [44, 160, 44]], alpha=1, linewidth=2.5, )
    # ax.plot(x0, y3, label='Smooth values',
    #        alpha=1, linewidth=2.5, )
    ax.plot(da_mdl['time_decimal'].values, da_mdl.values, label='original',
            marker='.', linestyle='none', color='gray', zorder=-10, alpha=0.2)
    ax.set_ylabel("$CO_2$ (ppm)")
    ax.set_xlabel("year")
    #
    plt.title(f'model [{opts.model_name}]')
    #
    aesthetic_grid_no_spines(ax)
    #
    plt.legend()
    #
    plt.tight_layout()
    #
    if opts.figure_savepath:
        mysavefig(fig=fig, plot_save_name=opts.figure_savepath + 'supplement1_mdl.png')

    # --- Make a supplemental figure for filter components ---

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

    # --- Do a seasonal climatology ---
    def make_cycle(x0, smooth_cycle):
        # Convert dates to datetime objects, and make a dataframe with a month column for grouping purposes.
        df_seasonalcycle = pd.DataFrame.from_dict({'datetime': [t2dt(i) for i in x0],
                                                   'co2': smooth_cycle})
        df_seasonalcycle['month'] = df_seasonalcycle['datetime'].dt.month

        # Bin by month, and add a column that represents months in datetime format for plotting purposes.
        df_monthly = df_seasonalcycle.groupby('month').mean().reset_index()
        df_monthly['month_datetime'] = pd.to_datetime(df_monthly['month'], format='%m')

        return df_monthly['month_datetime'], df_monthly['co2']
    #
    ref_dt, ref_vals = make_cycle(x0=filt_ref.xinterp,
                                 smooth_cycle=filt_ref.getHarmonicValue(filt_ref.xinterp) + filt_ref.smooth - filt_ref.trend)
    #
    mdl_dt, mdl_vals = make_cycle(x0=filt_mdl.xinterp,
               smooth_cycle=filt_mdl.getHarmonicValue(filt_mdl.xinterp) + filt_mdl.smooth - filt_mdl.trend)

    # --- Plot the seasonal cycle
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(ref_dt, ref_vals, label=f'obs [{opts.station_code}]', marker='o', color='k')
    ax.plot(mdl_dt, mdl_vals, label=f'model [{opts.model_name}]', marker='o', color='r')
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


def add_seasonal_cycle_args_to_parser(parser: argparse.PARSER) -> None:
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
    parser.add_argument('--globalmean', action='store_true')
    parser.add_argument('--use_mlo_for_detrending', action='store_true')
