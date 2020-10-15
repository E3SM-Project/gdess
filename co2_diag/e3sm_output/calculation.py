import numpy as np
import xarray as xr

from . import utils

import logging
logger = logging.getLogger(__name__)


def calc_change_in_mass(dataset, varname='glmean_TMCO2_FFF', prefix='deltas_'):
    """Change in mass of $CO_2$ from timestep to timestep
    $\Delta{mass_{CO_2}}$ - using backwards difference

    \begin{align*}
        && \Delta \texttt{glmean_TMCO2}_{t} = \texttt{glmean_TMCO2}_{t} - \texttt{glmean_TMCO2}_{t-1} && \forall t
    \end{align*}

    """
    dataset[prefix + varname] = xr.DataArray(calc_var_deltas(dataset[varname]),
                                             coords={'time': dataset['time']},
                                             dims=['time'])
    return dataset


def calc_global_weighted_means(dataset,
                               variable_list=None, prefix='glmean_', weighting_var='area_p', averaging_dims=('ncol')):
    """Global means with weighting by area

    xarray has recently introduced a weighting method (http://xarray.pydata.org/en/stable/examples/area_weighted_temperature.html).
    The area of each grid cell for the ne4pg2/ne30pg2 grids is called “area_p”.
    For the ne4/ne30 grids it is just called “area”.  Not sure why they changed it.

    \begin{align*}
        && \texttt{glmean_CO2var}_{t}= & \frac{\sum_{i=1}^{nlat}\sum_{j=1}^{nlon} \texttt{CO2var}_{i,j,t} * \texttt{area}_{i,j}}{\sum_{i=1}^{nlat}\sum_{j=1}^{nlon} \texttt{area}_{i,j}}     &&\forall t, \\
        with\,units: && \{kg/m^2\}_t= & \frac{\sum_{i=1}^{nlat}\sum_{j=1}^{nlon} \{kg/{m^2}\}_{i,j,t} * \{m^2/m^2\}_{i,j}}{\sum_{i=1}^{nlat}\sum_{j=1}^{nlon} \{m^2/m^2\}_{i,j}}     &&\forall t. \\
    \end{align*}

    *notes:*
    - we would need to multiply by the radius of Earth ($R^2_{E}$ in units of $\{m^2\}$) to get to $kg$ because `area` is in steradians rather than $m^2$.
    - the above multiplication by `area` ($m^2/m^2$) is taken care of in the xarray.DataArray.weighted() method.

    """
    dataset = utils.add_global_mean_vars(dataset, variable_list=variable_list, prefix=prefix,
                                         weighting_var=weighting_var, averaging_dims=averaging_dims)
    return dataset


def calc_time_integrated_fluxes(dataset,
                                prefix='timeint_'):
    """Time-integrated flux of $CO_2$ from surface emissions and aircraft
    $\int{flux_{CO_2}}$

    \begin{align*}
        && \int_{t-1}^{t} \texttt{SFCO2} \approx & \texttt{glmean_SFCO2}_{t} & * & \quad (time_t - time_{t-1})    &&\forall t, \\
        with\,units: && \{kg/m^2\}_t= & \{kg/m^2/s\}_t& * & \quad \{s\}_{t}    &&\forall t. \\
    \end{align*}
    """
    dtime = calc_time_deltas(dataset)
    #
    dataset[prefix + 'SFCO2'] = xr.DataArray(dataset['glmean_SFCO2'] * dtime,
                                             coords={'time': dataset['time']}, dims=['time'])
    #
    dataset[prefix + 'TAFCO2'] = xr.DataArray(dataset['glmean_TAFCO2'] * dtime,
                                             coords={'time': dataset['time']}, dims=['time'])
    #
    dataset[prefix + 'TOTALFLUX'] = xr.DataArray(dataset[prefix + 'SFCO2'] + dataset[prefix + 'TAFCO2'],
                                                 coords={'time': dataset['time']}, dims=['time'])

    return dataset


def calc_var_deltas(xr_da_):
    """Backward difference calculation"""
    return np.insert(np.diff(xr_da_), 0, 0)


def calc_time_deltas(xr_ds_):
    """Time deltas in seconds"""
    seconds_per_day = 24 * 60 * 60
    return (xr_ds_['time_bnds'].diff('nbnd') * seconds_per_day).astype('float').isel(nbnd=0).round()


def convert_co2_to_ppm(xr_ds_, co2_var_name='CO2'):
    """Convert E3SM $CO_2$ $kg/kg$ to $ppm$

    The CO2 variable of the input dataset is converted from kg/kg to ppm
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
