
def co2_molfrac_to_ppm(xr_ds_, co2_var_name='CO2'):
    """Convert Obspack $CO_2$ $mol/mol$ to $ppm$

    The CO2 variable of the input dataset is converted from mol/mol to ppm

    Where $CO_2$ is converted from units of $mol/mol$ to $ppm$ by:

    \begin{array}{lllll}
        ppmfac = & \times \frac{1e6\,mol_{air}}{1\,mol_{CO_2}}
    \end{array}

    \begin{align}
        CO_2\,dry\,air\,mole\,fraction\,(i.e. \frac{mol_{CO_2}}{mol_{air}}) \times ppmfac & = ppm \\
    \end{align}
    """
    ppmfac = 1e6

    temp_long_name = xr_ds_[co2_var_name].long_name

    # do the conversion
    xr_ds_[co2_var_name] = xr_ds_[co2_var_name]*ppmfac

    xr_ds_[co2_var_name].attrs["units"] = 'ppm'
    xr_ds_[co2_var_name].attrs['long_name'] = temp_long_name

    return xr_ds_


def co2_kgfrac_to_ppm(xr_ds_, co2_var_name='CO2'):
    """Convert E3SM $CO_2$ $kg/kg$ to $ppm$

    The CO2 variable of the input dataset is converted from kg/kg to ppm

    Where conversion from $CO_2$ $kg/kg$ to $ppm$ by:

    \begin{array}{lllll}
        CO_2\,dry-air\,mass\,fraction  & \times CO_2\,molar\,mass & \times dry-air\,molar\,mass & \times ppm \\
        \frac{kg_{CO_2}}{kg_{air}} & \times \frac{1\,kmol_{CO_2}}{44.01\,kg_{CO_2}} & \times \frac{28.9647\,kg_{air}}{1\,kmol_{air}} & \times \frac{1e6\,parts_{air}}{1\,part_{CO_2}}
    \end{array}
    """
    mwco2 = 44.01
    mwdry = 28.9647
    mwfac = mwdry / mwco2
    ppmfac = mwfac * 1e6

    temp_long_name = xr_ds_[co2_var_name].long_name

    # do the conversion
    xr_ds_[co2_var_name] = xr_ds_[co2_var_name]*ppmfac

    xr_ds_[co2_var_name].attrs["units"] = 'ppm'
    xr_ds_[co2_var_name].attrs['long_name'] = temp_long_name

    return xr_ds_
