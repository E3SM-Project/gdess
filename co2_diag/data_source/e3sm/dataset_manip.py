import xarray as xr

from co2_diag.data_operation_utils.geographic import get_closest_mdl_cell_dict


def latlon_select(xr_ds: xr.Dataset,
                  lat: float,
                  lon: float,
                  grid='native'
                  ) -> xr.Dataset:
    """Select from dataset the column that is closest to specified lat/lon pair

    Note: this is currently only implemented for the e3sm NATIVE grid

    Parameters
    ----------
    xr_ds
    lat
    lon
    grid

    Returns
    -------

    Raises
    ------

    """
    if grid == 'native':
        closest_mdl_point_dict = get_closest_mdl_cell_dict(xr_ds, lat=lat, lon=lon, coords_as_dimensions=False)
        return xr_ds.where(xr_ds['ncol'] == closest_mdl_point_dict['index'], drop=True)
    else:
        raise ValueError('Unexpected grid type <%s>, not implemented (yet)', grid)
