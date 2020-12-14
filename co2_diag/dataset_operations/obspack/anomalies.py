import numpy as np
import pandas as pd
import xarray as xr

import logging
_logger = logging.getLogger(__name__)


def daily_anomalies(dataset: xr.Dataset,
                    varname: str = 'value'
                    ) -> pd.DataFrame:
    varlist = ['time', varname]
    tempds = dataset[varlist]

    # Get daily values, and add additional temporal label coordinates
    tempds = tempds.resample(time="D").mean()
    tempds = tempds.assign_coords(doy=tempds.time.dt.strftime("%j").str.lstrip("0").astype(int))
    tempds = tempds.assign_coords(year=tempds.time.dt.strftime("%Y"))
    ds_daily = tempds.assign_coords(year_month=tempds.time.dt.strftime("%Y-%m"))

    # Convert to pandas dataframe
    df_daily = ds_daily.to_dataframe()

    # --- Calculate Anomalies ---
    #
    """
    Daily anomaly of this data from the local monthly mean of the time-series. 
    That is, I want to take away the average of (eg) January 1979 from all the days in January 1979. 
    And I'd like to do this for every month of every year in my array."""
    df_daily['daily_anomaly_from_month'] = \
        (ds_daily.groupby("year_month") - ds_daily.groupby("year_month").mean("time"))[varname]
    # Get each datum's departure from it's year's mean
    df_daily['daily_anomaly_from_year'] = (ds_daily.groupby("year") - ds_daily.groupby("year").mean("time"))[varname]
    # Get each datum's departure from the entire dataset's mean
    df_daily['daily_anomaly_from_allmean'] = (ds_daily - ds_daily.mean("time"))[varname]
    # Get each datum's departure from it's year's mean
    df_daily['daily_anomaly_from_year'] = (ds_daily.groupby("year") - ds_daily.groupby("year").mean("time"))[varname]

    df_daily = df_daily.reset_index()

    return df_daily


def monthly_anomalies(dataset: xr.Dataset,
                      varname: str = 'value'
                      ) -> pd.DataFrame:
    varlist = ['time', varname]
    tempds = dataset[varlist]

    # Get monthly values, and add additional temporal label coordinates
    tempds = tempds.resample(time="M").mean()
    tempds = tempds.assign_coords(moy=tempds.time.dt.strftime("%m"))
    tempds = tempds.assign_coords(year=tempds.time.dt.strftime("%Y"))
    ds_monthly = tempds.assign_coords(year_month=tempds.time.dt.strftime("%Y-%m"))

    # Convert to pandas dataframe
    df_monthly = ds_monthly.to_dataframe()

    # --- Calculate Anomalies ---
    #
    # Get each datum's departure from it's year's mean"""
    df_monthly['monthly_anomaly_from_year'] = \
        (ds_monthly.groupby("year") - ds_monthly.groupby("year").mean("time"))[varname]
    # Get each datum's departure from the entire dataset's mean"""
    df_monthly['monthly_anomaly_from_allmean'] = (ds_monthly - ds_monthly.mean("time"))[varname]

    df_monthly = df_monthly.reset_index()

    return df_monthly


def seasonal_anomalies(dataset: xr.Dataset,
                       varname: str = 'value'
                       ) -> pd.DataFrame:
    varlist = ['time', varname]
    tempds = dataset[varlist]

    # Get seasonal (quarterly, starting on December) values, and add additional temporal label coordinates
    tempds = tempds.resample(time="QS-DEC").mean()
    tempds = tempds.assign_coords(year=tempds.time.dt.strftime("%Y"))
    # add seasonal labels
    tempds = tempds.assign_coords(moy=tempds.time.dt.strftime("%m").astype(int))
    month_to_season_lu = np.array([
        None,
        'DJF', 'DJF',
        'MAM', 'MAM', 'MAM',
        'JJA', 'JJA', 'JJA',
        'SON', 'SON', 'SON',
        'DJF'
    ])
    tempds = tempds.assign_coords(season=('time', month_to_season_lu[tempds.moy]))
    ds_seasonal = tempds.assign_coords(year_month=tempds.time.dt.strftime("%Y-%m"))

    # Convert to pandas dataframe
    df_seasonal = ds_seasonal.to_dataframe()

    # --- Calculate Anomalies ---
    #
    # Get each datum's departure from the entire dataset's mean
    df_seasonal['seasonal_anomaly_from_allmean'] = (ds_seasonal - ds_seasonal.mean("time"))[varname]

    df_seasonal = df_seasonal.reset_index()

    return df_seasonal
