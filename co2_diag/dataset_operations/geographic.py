from math import cos, asin, sqrt

# Define functions to be imported by *, e.g. from the local __init__ file
#   (also to avoid adding above imports to other namespaces)
__all__ = ['distance', 'closest']


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))


def closest(data, v):
    min_entry = min(data, key=lambda p: distance(v['lat'], v['lon'], p['lat'], p['lon']))

    return min_entry


