import os
import numpy as np
import xarray as xr

from co2_diag.data_source.datasetdict import DatasetDict
from co2_diag.operations.time import ensure_dataset_datetime64
from co2_diag.operations.convert import co2_molfrac_to_ppm

import logging
_logger = logging.getLogger(__name__)


def dataset_from_filelist(file_list: list,
                          vars_to_keep: list = None,
                          decode_times: bool = False):
    """Load ObsPack NetCDF files specified in a list and create one Dataset from them.

    Parameters
    ----------
    file_list
    vars_to_keep: list
    decode_times: parameter passed to Xarray.open_dataset()

    Returns
    -------
    An xr.Dataset
    """
    if vars_to_keep is None:
        # These are the default variables to keep if not overridden by a passed parameter.
        vars_to_keep = ['value', 'nvalue', 'value_std_dev',
                        'time', 'start_time', 'datetime', 'time_decimal',
                        'latitude', 'longitude', 'altitude', 'pressure',
                        'qcflag', 'dataset_platform', 'dataset_project',
                        'obspack_num', 'obspack_id']

    ds_list = []
    for i, f in enumerate(file_list):
        thisds = xr.open_dataset(f, decode_times=decode_times)

        # If the following variables are not present, continue loading and just make them blank DataArrays
        #    Otherwise, we will raise an error
        possible_missing_vars = ['pressure', 'qcflag', 'value_std_dev', 'nvalue']
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


def load_data_with_regex(datadir: str,
                         compiled_regex_pattern=None,
                         ) -> DatasetDict:
    """Load into memory the data from regex-defined files of Globalview+.

    Parameters
    ----------
    datadir
        directory containing the Globalview+ NetCDF files.
    compiled_regex_pattern

    Returns
    -------
    dict
        Names, latitudes, longitudes, and altitudes of each station
    """
    # --- Go through files and extract all files found via the regex pattern search ---
    # file_dict = {s.group(1): f for f in os.listdir(datadir) if (s := compiled_regex_pattern.search(f))}
    file_dict = dict()
    for f in os.listdir(datadir):
        if s := compiled_regex_pattern.search(f):
            if s.group(1) not in file_dict.keys():
                file_dict[s.group(1)] = [f]
            else:
                file_dict[s.group(1)].append(f)
    _logger.debug('%s', '\n'.join([item for sublist in
                                   [[os.path.basename(ele) for ele in x]
                                    for x in file_dict.values()]
                                   for item in sublist]
                                  )
                  )

    ds_obs_dict = {}
    site_dict = {}
    for i, (sitecode, file_list) in enumerate(file_dict.items()):
        ds_obs_dict[sitecode] = dataset_from_filelist([os.path.join(datadir, f) for f in file_list])
        site_dict[sitecode] = {'name': ds_obs_dict[sitecode].site_name}

        lats = ds_obs_dict[sitecode]['latitude'].values
        lons = ds_obs_dict[sitecode]['longitude'].values
        # Get the latitude and longitude of each station
        #     different_station_lats = np.unique(lats)
        #     different_station_lons = np.unique(lons)
        # print(f"there are {len(different_station_lons)} different latitudes for the station: {different_station_lons}")

        # Get the average lat,lon
        meanlon = lons.mean()
        if meanlon < 0:
            meanlon = meanlon + 360
        SiteLatLon = {'lat': lats.mean(), 'lon': meanlon}
        _logger.info("%s. %s - %s", str(i).rjust(2), sitecode.ljust(12), SiteLatLon)

        site_dict[sitecode]['lat'] = lats.mean()
        site_dict[sitecode]['lon'] = meanlon

    # Wrangle -- Do the things to the Obs dataset.
    _logger.debug("Converting datetime format and units..")
    # Do the things to the Obs dataset.
    for k, v in ds_obs_dict.items():
        ds_obs_dict[k] = (v
                          .set_coords(['time', 'time_decimal', 'latitude', 'longitude', 'altitude'])
                          .sortby(['time'])
                          .swap_dims({"obs": "time"})
                          .pipe(ensure_dataset_datetime64)
                          .rename({'value': 'co2'})
                          .pipe(co2_molfrac_to_ppm, co2_var_name='co2')
                          .set_index(obs=['time', 'longitude', 'latitude', 'altitude'])
                          )
    #### Concatenate all sites into one large dataset, for mapping or other combined analysis purposes
    ds_all = xr.concat(ds_obs_dict.values(), dim=('obs'))

    return DatasetDict(ds_obs_dict)
