import numpy as np
import xarray as xr

import logging
logger = logging.getLogger(__name__)


def dataset_from_filelist(file_list: list,
                          vars_to_keep: list = None,
                          decode_times: bool = False):
    """Load ObsPack NetCDF files specified in a list and create one Dataset from them.

    :param file_list:
    :param vars_to_keep: list
    :param decode_times: parameter passed to Xarray.open_dataset()
    :return: xr.Dataset
    """
    if vars_to_keep is None:
        # These are the default variables to keep if not overridden by a passed parameter.
        vars_to_keep = ['value', 'nvalue', 'value_std_dev',
                        'time', 'start_time', 'datetime', 'time_decimal',
                        'latitude', 'longitude', 'altitude',
                        'qcflag', 'dataset_platform', 'dataset_project',
                        'obspack_num', 'obspack_id']

    ds_list = []
    for i, f in enumerate(file_list):
        thisds = xr.open_dataset(f, decode_times=decode_times)

        # If the following variables are not present, continue loading and just make them blank DataArrays
        #    Otherwise, we will raise an error
        possible_missing_vars = ['qcflag', 'value_std_dev', 'nvalue']
        for pmv in possible_missing_vars:
            if not (pmv in thisds.keys()):
                blankarray = xr.DataArray(data=[np.nan], dims='obs', name=pmv).squeeze()
                thisds = thisds.assign({pmv: blankarray})

        # Only the specified variables are retained.
        to_drop = []
        for vname in thisds.keys():
            if not (vname in vars_to_keep):
                to_drop.append(vname)
        newds = thisds.drop_vars(to_drop)

        # Dataset attributes 'platform' and 'project' are copied to every data point along the 'obs' dimension.
        n_obs = len(thisds['obs'])
        newds = newds.assign(dataset_platform=xr.DataArray([thisds.attrs['dataset_platform']] * n_obs, dims='obs'))
        newds = newds.assign(dataset_project=xr.DataArray([thisds.attrs['dataset_project']] * n_obs, dims='obs'))

        ds_list.append(newds)
        #     if i > 100:
        #         break

    ds = xr.concat(ds_list, dim='obs')

    return ds
