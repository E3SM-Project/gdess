import sys
import numpy as np
import xarray as xr

from co2_diag.nums import numstr

import logging
logFormatter = '%(message)s'  # %(asctime)s - %(levelname)s
logging.basicConfig(format=logFormatter, level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


def _change_log_level(level):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def by_decimalyear(dataset: xr.Dataset,
                   start: float = 2017, end: float = 2018,
                   verbose: bool = False) -> xr.Dataset:
    if verbose:
        _change_log_level(logging.INFO)

    # We start with the passed-in dataset.
    orig_shape = dataset['time_decimal'].shape
    keep_mask = np.full(orig_shape, True)
    logger.info(f"Original # data points: {numstr(orig_shape[0], 0)}")

    # The data are subsetted by year.
    keep_mask = keep_mask & (dataset['time_decimal'] >= start)
    keep_mask = keep_mask & (dataset['time_decimal'] < end)
    ds_year = dataset.where(keep_mask, drop=True)
    ds_year_shape = ds_year['time_decimal'].shape
    logger.info(f" -- subset between <start={start} and end={end}> -- # data points: {numstr(ds_year_shape[0], 0)}")

    return ds_year


def binLonLat(dataset: xr.Dataset,
              n_latitude: int = 10, n_longitude: int = 10):
    lon = dataset['longitude']
    lat = dataset['latitude']
    dat = dataset['value']

    # Data are binned onto the grid.
    #   (x & y must be reversed due to row-first indexing.)
    zi, y_edges, x_edges = np.histogram2d(lat.values, lon.values,
                                          bins=(n_latitude, n_longitude), weights=dat.values, normed=False)
    counts, _, _ = np.histogram2d(lat.values, lon.values,
                                  bins=(n_latitude, n_longitude))
    zi = np.ma.masked_equal(zi, 0)

    # Mean is calculated.
    zi = zi / counts
    zi = np.ma.masked_invalid(zi)

    return zi, y_edges, x_edges


def bin3d(dataset: xr.Dataset, vertical_bin_edges: np.ndarray,
          n_latitude: int = 10, n_longitude: int = 10, units: str = 'ppm',
          verbose: bool = True) -> xr.Dataset:
    if verbose:
        _change_log_level(logging.INFO)

    # We start with the passed-in dataset.
    orig_shape = dataset['time_decimal'].shape
    keep_mask_orig = np.full(orig_shape, True)

    # The vertical bins are defined.
    lvls = vertical_bin_edges
    n_vertical = len(lvls)

    # x_arr = []
    # y_arr = []
    value_arr = []
    lvl_pairs = []
    for i, (l0, l1) in enumerate(zip(lvls, lvls[1:])):
        if verbose:
            logger.info(f"-Vertical level {i+1}/{n_vertical-1}-")
        lvl_pairs.append([l0, l1])

        # The data are subsetted by altitude.
        keep_mask = keep_mask_orig & (dataset['altitude'] > l0)
        keep_mask = keep_mask & (dataset['altitude'] <= l1)
        if verbose:
            logger.info(f"  subset # data points: {numstr(np.count_nonzero(keep_mask), 0)}")

        # The data are binned along the x and y directions.
        values, y_edges, x_edges = binLonLat(dataset.where(keep_mask, drop=True),
                                             n_latitude=n_latitude, n_longitude=n_longitude)
        value_arr.append(values)

    # Horizontal centers of each bin are retrieved.
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
    vertical_centers = 0.5 * (vertical_bin_edges[1:] + vertical_bin_edges[:-1])

    # xg, yg = np.meshgrid(x_centers, y_centers)
    # x_arr.append(xg)
    # y_arr.append(yg)

    # The python lists are converted to 3d numpy arrays.
    z_arr = np.array(value_arr)
    # x_arr = np.array(x_arr)
    # y_arr = np.array(y_arr)

    logger.info(f"subset data shape: {z_arr.shape}")
    logger.info("\nDone.")

    ds_sub = xr.Dataset({
        'value': xr.DataArray(
            data=value_arr,
            dims=['vertical', 'lat', 'lon'],
            coords={'vertical': vertical_centers,
                    'lat': y_centers,
                    'lon': x_centers,
                    },
            attrs={
                '_FillValue': -999.9,
                'units': 'units'
            }
        ),
        'vertical_edges': xr.DataArray(data=[[l0, l1] for l0, l1 in zip(vertical_bin_edges, vertical_bin_edges[1:])],
                                       dims=['vertical', 'nbnds'],
                                       coords={'nbnds': [0, 1]}),
        'lat_edges': xr.DataArray(data=[[l0, l1] for l0, l1 in zip(y_edges, y_edges[1:])],
                                  dims=['lat', 'nbnds'],
                                  coords={'nbnds': [0, 1]}),
        'lon_edges': xr.DataArray(data=[[l0, l1] for l0, l1 in zip(x_edges, x_edges[1:])],
                                  dims=['lon', 'nbnds'],
                                  coords={'nbnds': [0, 1]}),
    }
    )

    return ds_sub

# def bin_multiyear(dd)