import xarray as xr

from .calculation import get_closest_mdl_cell_dict


def latlon_select(xr_ds: xr.Dataset,
                  lat: float,
                  lon: float,
                  grid='native'
                  ) -> xr.Dataset:
    """Select from dataset the column that is closest to specified lat/lon pair

    Parameters
    ----------
    xr_ds
    lat
    lon
    grid

    Returns
    -------

    """
    closest_mdl_point_dict = get_closest_mdl_cell_dict(xr_ds, lat=lat, lon=lon)

    if grid == 'native':
        return xr_ds.where(xr_ds['ncol'] == closest_mdl_point_dict['index'], drop=True)
    else:
        raise ValueError('Unexpected grid type <%s>, not implemented (yet)', grid)
