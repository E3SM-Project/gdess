from gdess import load_stations_dict
from typing import Union
import logging

_logger = logging.getLogger(__name__)


def populate_station_list(run_all_stations: bool,
                          station_list: Union[list, str]) -> list:
    """The list of stations to analyze is populated.

    Parameters
    ----------
    run_all_stations : bool
    station_list : Union[list, str]

    Returns
    -------
    list
    """
    if run_all_stations:
        stations_dict = load_stations_dict()
        stations_to_analyze = list(stations_dict.keys())
    elif station_list:
        stations_to_analyze = station_list
    else:
        raise ValueError('Unexpected empty station list')

    return stations_to_analyze


