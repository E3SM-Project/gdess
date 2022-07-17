import cftime
import logging
import datetime as pydt
from datetime import timedelta
from typing import Sequence, Union, List

import numpy as np
import pandas as pd
import xarray as xr

_logger = logging.getLogger(__name__)


def ensure_dataset_datetime64(dataset: xr.Dataset
                              ) -> xr.Dataset:
    """Often it is more convenient to work with the `time` variable as type `datetime64`.

    Parameters
    ----------
    dataset : ``xarray.Dataset``

    Returns
    -------
    ``xarray.Dataset``
    """
    dataset = xr.decode_cf(dataset)
    dataset['time'] = ensure_datetime64_array(dataset['time'])

    return dataset


def year_to_datetime64(yr: str) -> Union[None, np.datetime64]:
    if yr is not None:
        return np.datetime64(yr, 'D')
    else:
        return None


def to_datetimeindex(dataset: xr.Dataset) -> xr.Dataset:
    """It is often more convenient to work with the `time` variable as type `datetime64`.

    Parameters
    ----------
    dataset : ``xarray.Dataset``

    Returns
    -------
    ``xarray.Dataset``
    """
    # Check if it's already a datetimeindex
    if isinstance(dataset.indexes['time'], pd.core.indexes.datetimes.DatetimeIndex):
        _logger.debug('already a datetimeindex, no conversion done.')
    else:
        dataset['time'] = dataset.indexes['time'].to_datetimeindex()
        return dataset


def ensure_datetime64_array(time: Sequence) -> Sequence:
    """Convert an input 1D array to an array of numpy.datetime64 objects

    Parameters
    ----------
    time: Sequence

    Raises
    ------
    ValueError
        if unable to cast array to numpy.datetime64

    Returns
    -------
    Sequence
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


def ensure_cftime_array(time: Sequence) -> Sequence:
    """Convert an input 1D array to an array of cftime objects.

    Parameters
    ----------
    time : `Sequence`

    Returns
    -------
    `Sequence`
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
                   timestart: Union[cftime.DatetimeGregorian, np.datetime64],
                   timeend: Union[cftime.DatetimeGregorian, np.datetime64],
                   varlist: List[str] = None,
                   drop: bool = True,
                   drop_dups: bool = True
                   ) -> xr.Dataset:
    """Select part of a dataset between two times

    Parameters
    ----------
    dataset : ``xarray.Dataset``
    timestart : ``cftime.DatetimeGregorian`` or ``numpy.datetime64``
        must be of appropriate type for comparison with dataset.time type
        (e.g. cftime.DatetimeGregorian or numpy.datetime64)
    timeend : ``cftime.DatetimeGregorian`` or ``numpy.datetime64``
        must be of appropriate type for comparison with dataset.time type
        (e.g. cftime.DatetimeGregorian or numpy.datetime64)
    varlist : `list` of `str`
    drop : `bool`, default `True`
    drop_dups : `bool`, default `True`

    Returns
    -------
    ``xarray.Dataset``
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


def monthlist(dates: list) -> list:
    """Generate a list of months between two dates

    Parameters
    ----------
    dates : `list`
        Of length==2, with a start and end date, in the format of "%Y-%m-%d"

    Returns
    -------
    `list`
        Contains months (as numpy.datetime64 objects for the first day of each month)

    Example
    _______
    >>> monthlist_fast(['2017-01-01', '2017-04-01'])
    [numpy.datetime64('2017-01-01'),
     numpy.datetime64('2017-02-01'),
     numpy.datetime64('2017-03-01'),
     numpy.datetime64('2017-04-01')]
    """
    start, end = [pydt.datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    def total_months(dt): return dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(np.datetime64(pydt.datetime(y, m+1, 1).strftime("%Y-%m"), 'D'))

    return mlist


def dt2t(year: int, month: int, day: int,
         h: int = 0, m: int = 0, s: int = 0) -> float:
    """Convert parts of a DT.datetime to a float

    Parameters
    ----------
    year : `int`
    month : `int`
    day : `int`
    h : `int`
    m : `int`
    s : `int`

    Returns
    -------
    `float`
    """
    year_seconds = (pydt.datetime(year, 12, 31, 23, 59, 59, 999999) -
                    pydt.datetime(year, 1, 1, 0, 0, 0)).total_seconds()
    second_of_year = (pydt.datetime(year, month, day, h, m, s) -
                      pydt.datetime(year, 1, 1, 0, 0, 0)).total_seconds()
    return year + second_of_year / year_seconds


def t2dt(atime: float) -> pydt.datetime:
    """Convert a time (a float) to DT.datetime

    This is the inverse of dt2t, i.e.
        assert dt2t(t2dt(atime)) == atime

    Parameters
    ----------
    atime : `float`

    Returns
    -------
    datetime.datetime
    """
    year = int(atime)
    remainder = atime - year
    boy = pydt.datetime(year, 1, 1)
    eoy = pydt.datetime(year + 1, 1, 1)
    seconds = remainder * (eoy - boy).total_seconds()
    return boy + timedelta(seconds=seconds)
