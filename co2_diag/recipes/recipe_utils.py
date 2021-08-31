from typing import Union

from ccgcrv.ccg_dates import decimalDateFromDatetime
from co2_diag import load_stations_dict
from co2_diag.data_source.models.cmip.cmip_collection import Collection as cmipCollection
import pandas as pd
import xarray as xr
import logging

_logger = logging.getLogger(__name__)


def load_cmip_model_output(model_name: str,
                           cmip_load_method: str,
                           verbose=True) -> (bool, xr.Dataset):
    """Load CMIP model output

    We will only compare against CMIP model outputs if a model_name is supplied, otherwise return dataset as None.

    Parameters
    ----------
    model_name
    cmip_load_method
    verbose

    Returns
    -------

    """
    if compare_against_model := bool(model_name):
        _logger.info('*Processing CMIP model output*')
        new_self, _ = cmipCollection._recipe_base(datastore='cmip6', verbose=verbose, model_name=model_name,
                                                  load_method=cmip_load_method, skip_selections=True,
                                                  pickle_file=None)
        ds_mdl = new_self.stepB_preprocessed_datasets[model_name]
        ds_mdl = ds_mdl.assign_coords(time_decimal=('time', [decimalDateFromDatetime(x)
                                                             for x in pd.DatetimeIndex(ds_mdl['time'].values)]))
    else:
        ds_mdl = None
    return compare_against_model, ds_mdl


def populate_station_list(run_all_stations: bool,
                          station_list: Union[bool, list, str]) -> list:
    """The list of stations to analyze is populated.

    Parameters
    ----------
    run_all_stations
    station_list

    Returns
    -------

    """
    if run_all_stations:
        stations_dict = load_stations_dict()
        stations_to_analyze = stations_dict.keys()
    elif station_list:
        stations_to_analyze = station_list
    else:
        raise ValueError('Unexpected empty station list')

    return stations_to_analyze
