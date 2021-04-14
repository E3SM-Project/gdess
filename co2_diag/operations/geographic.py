from math import cos, asin, sqrt

# Define functions to be imported by *, e.g. from the local __init__ file
#   (also to avoid adding above imports to other namespaces)
__all__ = ['distance', 'closest', 'get_closest_mdl_cell_dict']

import xarray as xr


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))


def closest(data, v):
    min_entry = min(data, key=lambda p: distance(v['lat'], v['lon'], p['lat'], p['lon']))

    return min_entry


def get_closest_mdl_cell_dict(dataset: xr.Dataset,
                              lat: float,
                              lon: float,
                              coords_as_dimensions: bool = True
                              ) -> dict:
    """Find the nearest point in the model output

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
    A dict with lat, lon, and index in Dataset
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
