from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import logging
_logger = logging.getLogger(__name__)


def to_datetime64(dataset: xr.Dataset
                  ) -> xr.Dataset:
    # It is more convenient to work with the `time` variable as type `datetime64`.
    return xr.decode_cf(dataset)


def to_datetimeindex(dataset: xr.Dataset
                     ) -> xr.Dataset:
    # For model output, it is often more convenient to work with the `time` variable as type `datetime64`.

    # Check if it's already a datetimeindex
    if isinstance(dataset.indexes['time'], pd.core.indexes.datetimes.DatetimeIndex):
        _logger.debug('already a datetimeindex, no conversion done.')
    else:
        dataset['time'] = dataset.indexes['time'].to_datetimeindex()
        return dataset


def select_between(dataset: xr.Dataset,
                   timestart: np.datetime64,
                   timeend: np.datetime64,
                   varlist=None,
                   drop_dups=True) -> xr.Dataset:
    """Select part of a dataset between two times

    Parameters
    ----------
    dataset
    timestart
    timeend
    varlist
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

    return ds_sub.where(tempmask, drop=True)


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
