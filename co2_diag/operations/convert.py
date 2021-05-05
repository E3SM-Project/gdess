import xarray as xr

import logging
_logger = logging.getLogger(__name__)


def co2_molfrac_to_ppm(xr_ds_: xr.Dataset,
                       co2_var_name: str = 'co2',
                       ) -> xr.Dataset:
    """Convert CO2 from units of mol/mol to ppm

    Parameters
    ----------
    xr_ds_
        A dataset containing a CO2 variable
    co2_var_name
        Variable name for CO2 in the dataset

    Returns
    -------
        The dataset with CO2 converted to units of <ppm>

    Notes
    _______
    The $CO_2$ variable of the input dataset is converted from units of $mol/mol$ to $ppm$ by:

    \begin{array}{lllll}
        ppmfac = & \times \frac{1e6\,mol_{air}}{1\,mol_{CO_2}}
    \end{array}

    \begin{align}
        CO_2\,dry\,air\,mole\,fraction\,(i.e. \frac{mol_{CO_2}}{mol_{air}}) \times ppmfac & = ppm \\
    \end{align}
    """
    ppmfac = 1e6

    temp_long_name = xr_ds_[co2_var_name].long_name
    _logger.debug("\toriginal units <%s>", xr_ds_[co2_var_name].attrs['units'])

    # do the conversion
    xr_ds_[co2_var_name] = xr_ds_[co2_var_name]*ppmfac

    xr_ds_[co2_var_name].attrs["units"] = 'ppm'
    xr_ds_[co2_var_name].attrs['long_name'] = temp_long_name

    _logger.debug("\tnew units <%s>", xr_ds_[co2_var_name].attrs['units'])

    return xr_ds_


def co2_kgfrac_to_ppm(xr_ds_: xr.Dataset,
                      co2_var_name: str = 'co2',
                      ) -> xr.Dataset:
    """Convert CO2 from units of kg/kg to ppm

    Parameters
    ----------
    xr_ds_
        A dataset containing a CO2 variable
    co2_var_name
        Variable name for CO2 in the dataset

    Returns
    -------
        The dataset with CO2 converted to units of <ppm>

    Notes
    ______
    The $CO_2$ variable of the input dataset is converted from units of $kg/kg$ to $ppm$ by:

    \begin{array}{lllll}
        CO_2\,dry-air\,mass\,fraction  & \times CO_2\,molar\,mass & \times dry-air\,molar\,mass & \times ppm \\
        \frac{kg_{CO_2}}{kg_{air}} & \times \frac{1\,kmol_{CO_2}}{44.01\,kg_{CO_2}} & \times \frac{28.9647\,kg_{air}}{1\,kmol_{air}} & \times \frac{1e6\,parts_{air}}{1\,part_{CO_2}}
    \end{array}
    """
    mwco2 = 44.01
    mwdry = 28.9647
    mwfac = mwdry / mwco2
    ppmfac = mwfac * 1e6

    temp_long_name = ''
    if 'long_name' in xr_ds_[co2_var_name].attrs:
        temp_long_name = xr_ds_[co2_var_name].long_name

    # do the conversion
    xr_ds_[co2_var_name] = xr_ds_[co2_var_name]*ppmfac

    xr_ds_[co2_var_name].attrs["units"] = 'ppm'
    xr_ds_[co2_var_name].attrs['long_name'] = temp_long_name

    return xr_ds_
