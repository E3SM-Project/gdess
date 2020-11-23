import numpy as np
import xarray as xr


def to_datetime64(dataset: xr.Dataset
                  ) -> xr.Dataset:
    # It is more convenient to work with the `time` variable as type `datetime64`.
    return xr.decode_cf(dataset)


def to_datetimeindex(dataset: xr.Dataset
                     ) -> xr.Dataset:
    # For model output, it is often more convenient to work with the `time` variable as type `datetime64`.
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
