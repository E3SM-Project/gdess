from math import cos, asin, sqrt

import xarray as xr

# Define functions to be imported by *, e.g. from the local __init__ file
#   (also to avoid adding above imports to other namespaces)
__all__ = ['distance', 'closest', 'get_closest_mdl_cell_dict']


def distance(lat1: float,
             lon1: float,
             lat2: float,
             lon2: float) -> float:
    p = 0.017453292519943295
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))


def closest(data: list,
            v: dict) -> dict:
    """Find closest point in `data` to the v point

    Parameters
    ----------
    data
    v

    Returns
    -------
    dict
    """
    min_entry = min(data, key=lambda p: distance(v['lat'], v['lon'], p['lat'], p['lon']))

    return min_entry


def get_closest_mdl_cell_dict(dataset: xr.Dataset,
                              lat: float,
                              lon: float,
                              coords_as_dimensions: bool = True
                              ) -> dict:
    """Find the point in the model output that is closest to specified lat/lon pair

    Examples
    --------
    To get the data subset at a location:
    -- For CMIP outputs --
        >>> closest_point = get_closest_mdl_cell_dict(dataset, lat=24.3, lon=137.8, coords_as_dimensions=True)
        >>> dataset.stack(coord_pair=['lat', 'lon']).isel(coord_pair=closest_point['index'])
    -- For E3SM native grid --
        >>> closest_mdl_point = get_closest_mdl_cell_dict(dataset, lat=lat, lon=lon, coords_as_dimensions=False)
        >>> dataset.where(dataset['ncol'] == closest_mdl_point['index'], drop=True)

    Parameters
    ----------
    dataset
    lat
    lon
    coords_as_dimensions
        True for dataset variables as independent dimensions, e.g.
            lat = -90, -89,...,0, 1,... ,90
            lon = -180, -179, -178,...,0, 1,... ,180
        False for dataset variables where all pairs are enumerated, e.g.
            lat = -90,  -90,...  ,-89,  -89,... 90, 90
            lon = -180, -179,... ,-180, -179,... 179, 180

    Returns
    -------
    dict
        With lat, lon, and index in Dataset
        For example, {'lat': 19.5, 'lon': 204.375, 'index': 31555}
    """
    obs_station_lat_lon = {'lat': lat, 'lon': lon}

    if coords_as_dimensions:
        coords = dataset.stack(coord_pair=['lat', 'lon']).coord_pair.values
    else:
        coords = zip(dataset['lat'].values, dataset['lon'].values)

    mdl_lat_lon_list = [{'lat': a, 'lon': o, 'index': i}
                        for i, (a, o)
                        in enumerate(coords)]

    # Find it.
    closest_dict = closest(mdl_lat_lon_list, obs_station_lat_lon)

    return closest_dict
