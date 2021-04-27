from datetime import datetime

import datetime as pydt
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
import cftime

import logging
_logger = logging.getLogger(__name__)


def ensure_dataset_datetime64(dataset: xr.Dataset
                              ) -> xr.Dataset:
    # It is more convenient to work with the `time` variable as type `datetime64`.
    dataset = xr.decode_cf(dataset)
    dataset['time'] = ensure_datetime64_array(dataset['time'])

    return dataset


def year_to_datetime64(yr: str):
    if yr is not None:
        return np.datetime64(yr, 'D')
    else:
        return None


def to_datetimeindex(dataset: xr.Dataset
                     ) -> xr.Dataset:
    # For model output, it is often more convenient to work with the `time` variable as type `datetime64`.

    # Check if it's already a datetimeindex
    if isinstance(dataset.indexes['time'], pd.core.indexes.datetimes.DatetimeIndex):
        _logger.debug('already a datetimeindex, no conversion done.')
    else:
        dataset['time'] = dataset.indexes['time'].to_datetimeindex()
        return dataset


def ensure_datetime64_array(time: Sequence):
    """Convert an input 1D array to an array of numpy.datetime64 objects.
    """
    if isinstance(time, xr.DataArray):
        time = time.indexes["time"]
    elif isinstance(time, np.ndarray):
        time = pd.DatetimeIndex(time)
    if isinstance(time[0], np.datetime64):
        return time
    if isinstance(time[0], pydt.datetime) | isinstance(time[0], cftime.datetime):
        return np.array(
            [np.datetime64(ele) for ele in time]
        )
    raise ValueError("Unable to cast array to numpy.datetime64 dtype")


def ensure_cftime_array(time: Sequence):
    """Convert an input 1D array to an array of cftime objects.

    Parameters
    ----------
    time

    Returns
    -------
    Python's datetime are converted to cftime.DatetimeGregorian.

    Raises
    ------
    ValueError when unable to cast the input.
    """
    if isinstance(time, xr.DataArray):
        time = time.indexes["time"]
    elif isinstance(time, np.ndarray):
        time = pd.DatetimeIndex(time)
    if isinstance(time[0], cftime.datetime):
        return time
    if isinstance(time[0], pydt.datetime):
        return np.array(
            [cftime.DatetimeGregorian(*ele.timetuple()[:6]) for ele in time]
        )
    raise ValueError("Unable to cast array to cftime dtype")


def ensure_dataset_cftime(dataset):
    dataset['time'] = ensure_cftime_array(dataset['time'])
    return dataset


def select_between(dataset: xr.Dataset,
                   timestart,
                   timeend,
                   varlist=None,
                   drop=True,
                   drop_dups=True) -> xr.Dataset:
    """Select part of a dataset between two times

    Parameters
    ----------
    dataset
    timestart
        must be of appropriate type for comparison with dataset.time type
        (e.g. cftime.DatetimeGregorian or numpy.datetime64)
    timeend
        must be of appropriate type for comparison with dataset.time type
        (e.g. cftime.DatetimeGregorian or numpy.datetime64)
    varlist
    drop
        True (default) / False
    drop_dups
        True (default) / False

    Returns
    -------
    a subset of the original dataset with only times between timestart and timeend

    """
    if varlist is None:
        ds_sub = dataset.copy()
    else:
        ds_sub = dataset[varlist].copy()

    # Drop duplicate times
    if drop_dups:
        _, index = np.unique(ds_sub['time'], return_index=True)
        ds_sub = ds_sub.isel(time=index)

    # Select a time period
    tempmask = ds_sub['time'] >= timestart
    tempmask = tempmask & (ds_sub['time'] <= timeend)

    return ds_sub.where(tempmask, drop=drop)


def monthlist(dates) -> list:
    """Generate a list of months between two dates

    Parameters
    ----------
    dates
        A list of length=2, with a start and end date, in the format of "%Y-%m-%d"

    Returns
    -------
    A list containing months (as numpy.datetime64 objects for the first day of each month)

    Example
    _______

    >>> monthlist_fast(['2017-01-01', '2017-04-01'])
    [numpy.datetime64('2017-01-01'),
     numpy.datetime64('2017-02-01'),
     numpy.datetime64('2017-03-01'),
     numpy.datetime64('2017-04-01')]
    """
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    def total_months(dt): return dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(np.datetime64(datetime(y, m+1, 1).strftime("%Y-%m"), 'D'))

    return mlist
