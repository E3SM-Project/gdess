import numpy as np
import xarray as xr

import logging

_logger = logging.getLogger(__name__)

# Define functions to be imported by *, e.g. from the local __init__ file
#   (also to avoid adding above imports to other namespaces)
__all__ = ['calc_change_in_mass',
           'calc_global_weighted_means', 'add_global_mean_vars',
           'calc_time_integrated_fluxes',
           'calc_var_deltas', 'calc_time_deltas']


def calc_change_in_mass(dataset: xr.Dataset,
                        varname='glmean_TMCO2_FFF', prefix='deltas_'
                        ) -> xr.Dataset:
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


def calc_global_weighted_means(dataset: xr.Dataset,
                               variable_list=None,
                               prefix='glmean_', weighting_var='area_p', averaging_dims=('ncol')
                               ) -> xr.Dataset:
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
    dataset = add_global_mean_vars(dataset, variable_list=variable_list, prefix=prefix,
                                   weighting_var=weighting_var, averaging_dims=averaging_dims)
    return dataset


def calc_time_integrated_fluxes(dataset: xr.Dataset,
                                prefix='timeint_'
                                ) -> xr.Dataset:
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


def calc_var_deltas(xr_da_: xr.DataArray
                    ) -> np.ndarray:
    """Backward difference calculation
    """
    return np.insert(np.diff(xr_da_), 0, 0)


def calc_time_deltas(xr_ds_: xr.Dataset
                     ) -> xr.DataArray:
    """Time deltas in seconds
    """
    seconds_per_day = 24 * 60 * 60
    return (xr_ds_['time_bnds'].diff('nbnd') * seconds_per_day).astype('float').isel(nbnd=0).round()


def add_global_mean_vars(xr_ds_: xr.Dataset,
                         variable_list,
                         prefix='glmean_', weighting_var='area_p', averaging_dims=('ncol')
                         ) -> xr.Dataset:
    """ For each variable in the input list, we add an accompanying variable that represents a global mean
    """
    for var in variable_list:
        xr_ds_[prefix + var] = xr_ds_[var].weighted(xr_ds_[weighting_var]).mean(averaging_dims)
        xr_ds_[prefix + var].attrs["units"] = xr_ds_[var].units
        xr_ds_[prefix + var].attrs['long_name'] = xr_ds_[var].long_name + ' (globally averaged)'

    return xr_ds_


def getPMID(hyam, hybm, P0, PS):
    PMID = P0 * hyam + PS * hybm
    PMID.attrs.update({'units': 'Pa',
                       'long_name': 'Pressure'})
    return PMID


def getPINT(hyai, hybi, P0, PS):
    PMID = P0 * hyai + PS * hybi
    PMID.attrs.update({'units': 'Pa',
                       'long_name': 'Pressure at interface levels'})
    return PMID
