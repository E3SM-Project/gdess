# -*- coding: utf-8 -*-
"""A collection of tools for use with the CO2 diagnostics development
Most of the routines are designed to work with xarray.DataArray types

Created September 2020
@author: Daniel E. Kaufman
"""
import os
import logging
from typing import Union, Sequence

import pandas as pd
import xarray as xr

_logger = logging.getLogger(__name__)

# Define functions to be imported by *, e.g. from the local __init__ file
#   (also to avoid adding above imports to other namespaces)
__all__ = ['print_var_summary', 'get_var_stats']


def where_am_i() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def pipe_df_head(dataf: pd.DataFrame, n_rows: int = 5):
    print(f"print_dataf: {dataf.head(n_rows)}")
    return dataf


def print_var_summary(dataset: xr.Dataset,
                      varname: str = 'CO2',
                      return_dataset: bool = False
                      ) -> Union[None, xr.Dataset]:
    """Brief stats for a dataset variable are printed

    Parameters
    ----------
    dataset : xarray.Dataset
    varname : str, default 'CO2'
    return_dataset : bool, default False

    Returns
    -------
    Either None or an xarray.Dataset
    """
    # We check if there are units specified for this variable
    vu = None
    if 'units' in dataset[varname].attrs:
        vu = dataset[varname].units
    # We check if there is a long name specified for this variable
    ln = None
    if 'long_name' in dataset[varname].attrs:
        ln = dataset[varname].long_name

    stats_dict = get_var_stats(dataset[varname])

    _logger.info("Summary for <%s>%s%s:",
                 varname,
                 ' (units of ' + vu + ')' if vu else '',
                 ' (long_name: ' + ln + ')' if ln else '')
    _logger.info("  min: %s", str(stats_dict['min']))
    _logger.info("  mean: %s", str(stats_dict['mean']))
    _logger.info("  max: %s", str(stats_dict['max']))

    my_shape = dataset[varname].shape
    dim_strings = [f"{d}: {my_shape[i]}" for i, d in enumerate(dataset[varname].dims)]
    _logger.info("  shape: (" + ', '.join(dim_strings) + ")")

    if return_dataset:
        return dataset


def assert_expected_dimensions(data: Union[xr.Dataset, xr.DataArray],
                               expected_dims: Sequence[str],
                               optional_dims: Sequence[str] = None,
                               expected_shape: Union[dict, list] = None
                               ) -> bool:
    """Raise an AssertionError if data dimensions don't match the given names or shape

    Note
    ----
    If an expected_shape argument isn't provided, we ignore the shapes (dim lengths).

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
    expected_dims : Sequence[str]
        names of the expected dimensions
    optional_dims : Sequence[str] (Optional)
        names of dimensions that will not raise an error whether they are present or not
    expected_shape : list or dict (Optional)
        Expected lengths for each dimension
        Note: If a list is provided, the order must match the dimension order of the expected_dims argument.

    Raises
    ------
    AssertionError, if the data dimensions or shape don't match those given.
    TypeError, if the arguments types are incorrect.
    ValueError, if the given shape and dimension arguments don't match.

    Returns
    -------
    bool
        True, if the given names (and shapes, if given) match the data.
    """
    dims_dict = dict(data.dims)

    # Names of the data dimensions are checked against the expected names.
    # Optional names are discarded before deciding whether to raise an AssertionError.
    missing_from_data = set(expected_dims) - set(dims_dict)
    missing_from_expected = set(dims_dict) - set(expected_dims)
    if optional_dims:
        for d in optional_dims:
            missing_from_data.discard(d)
            missing_from_expected.discard(d)
    msg = ""
    if missing_from_expected:
        msg += f"Dimesions {missing_from_expected} are in the data, but were not expected. "
    if missing_from_data:
        msg += f"Dimesions {missing_from_data} were expected, but are not in the data. "
    if msg:
        raise AssertionError(msg)

    # Lengths of the data dimensions are checked.
    if expected_shape:
        if len(expected_shape) != len(expected_dims):
            raise ValueError("Expected dimensions and shape must match.")

        expected_shapes_dict = expected_shape
        if isinstance(expected_shape, dict):
            pass
        elif isinstance(expected_shape, list):
            # For a list, we assume that the list order matches the order given in the expected_dims argument.
            expected_shapes_dict = {k: v for k, v in zip(expected_dims, expected_shape)}
        else:
            raise TypeError("The expected_shape argument should be list or dict. A <%s> was provided."
                            % type(expected_shape))
        shapes_match = all((v == expected_shapes_dict[k]) for k, v in dims_dict.items())

        if not shapes_match:
            msg = f"Dimension lengths in the data are {dims_dict}, not the expected {expected_shapes_dict}"
            raise AssertionError(msg)

    return True


def get_var_stats(dataarray: xr.DataArray) -> dict:
    """Retrieve a dictionary with summary statistics

    Parameters
    ----------
    dataarray : xarray.Dataarray

    Returns
    -------
    dict
    """
    return {
        'min': dataarray.min().values.item(),
        'mean': dataarray.mean().values.item(),
        'max': dataarray.max().values.item()
    }
