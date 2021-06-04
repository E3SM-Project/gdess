import re
from typing import Union

import numpy as np
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

import cartopy.crs as ccrs

from co2_diag import set_verbose
import co2_diag.data_source as co2ops
from co2_diag.data_source.obspack.load import load_data_with_regex
from co2_diag.data_source.multiset import Multiset
from co2_diag.recipes.utils import benchmark_recipe
from co2_diag.data_source.datasetdict import DatasetDict
from co2_diag.graphics.mapping import make_my_base_map

import logging
_loader_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))


class Collection(Multiset):
    def __init__(self, verbose: Union[bool, str] = False):
        """Instantiate a Obspack Aircraft Collection object.

        Parameters
        ----------
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        set_verbose(_loader_logger, verbose)

        self.df_combined_and_resampled = None
        # Define the stations that will be included in the dataset and available for diagnostic plots
        # self.station_dict = {'mlo': {'name': 'Mauna Loa'},
        #                      'brw': {'name': 'Barrow'},
        #                      'spo': {'name': 'South Pole Observatory'},
        #                      'smo': {'name': 'American Samoa'},
        #                      'zep': {'name': 'Zeppelin Observatory'},
        #                      'psa': {'name': 'Palmer Station'}}
        # _loader_logger.info("Loading data for %d observing stations..", len(self.station_dict))

        super().__init__(verbose=verbose)

    @staticmethod
    def prep_aircraft_site_data(dataset: xr.Dataset,
                                startdate: np.datetime64,
                                enddate: np.datetime64,
                                altitude_range: tuple
                                ) -> Union[dict, None]:
        """ Average the data for the time period specified, and between the altitudes specified
        """
        # -- a time range
        temp_ds = co2ops.obspack.subset.by_datetime(dataset, verbose='WARN',
                                                    start=startdate, end=enddate)
        if not temp_ds:
            return None

        # -- non-negative values
        keep_mask = np.full(temp_ds['co2'].shape, True)
        keep_mask = keep_mask & (temp_ds['co2'] > 0)
        # -- an altitude range
        keep_mask = keep_mask & (temp_ds['altitude'] <= altitude_range[1]) & (temp_ds['altitude'] > altitude_range[0])
        #
        lat = temp_ds['latitude'].where(keep_mask).mean(dim=('obs')).values.item()
        lon = temp_ds['longitude'].where(keep_mask).mean(dim=('obs')).values.item()
        temp_data = temp_ds['co2'].where(keep_mask).mean(dim=('obs'))
        temp_data = np.ma.masked_equal(temp_data, 0).data

        if temp_data.size != 1:
            raise ValueError('unexpected result of averaging.')

        return {'value': float(temp_data), 'lat': lat, 'lon': lon}

    @staticmethod
    def _load_aircraft_data(datadir: str,
                            ) -> DatasetDict:
        """Load into memory the data for aircraft measurements from Globalview+.

        Parameters
        ----------
        datadir: str
            directory containing the Globalview+ NetCDF files.

        Returns
        -------
        dict
            Names, latitudes, longitudes, and altitudes of each station
        """
        # --- Go through files and extract all 'aircraft' sampled files ---
        p = re.compile(r'co2_([a-zA-Z0-9]*)_aircraft.*\.nc$')
        return_value = load_data_with_regex(datadir=datadir, compiled_regex_pattern=p)

        return return_value

    @staticmethod
    def plot_aircraft_co2_map(longitudes: xr. DataArray,
                              latitudes: xr.DataArray,
                              values: xr.DataArray,
                              cmap: matplotlib.colors.ListedColormap,
                              cnorm: matplotlib.colors.BoundaryNorm,
                              titlestr: str = ''
                              ) -> (plt.Figure, plt.Axes):
        """Make a map of co2 concentrations from a dataarray

        Parameters
        ----------
        longitudes
        latitudes
        values
        cmap
        cnorm
        titlestr

        Returns
        -------

        """
        # A figure is created.
        coastline_kw = {'color': 'gray',
                        'linewidth': 0.1}
        borders_kw = {'color': 'gray',
                      'linestyle': ':'}
        fig, ax = make_my_base_map(coastline_kw=coastline_kw, borders_kw=borders_kw)

        # Plot the data
        pc = ax.scatter(longitudes, latitudes, c=values, s=100, alpha=0.8,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap, norm=cnorm,
                        edgecolors=None, linestyle=':', linewidth=0.2)
        #
        # Figure extras
        cb = fig.colorbar(pc, shrink=.8, pad=0.05, orientation='horizontal')
        cb.solids.set(alpha=1)
        cb.set_label('ppm')
        #
        ax.set_title(titlestr)

        return fig, ax

    @classmethod
    def plot_site_map(cls, site_dataset, time_range, altitude_range):
        # -- a time range
        temp_ds = co2ops.obspack.subset.by_datetime(site_dataset, verbose='WARN',
                                                    start=time_range[0], end=time_range[1])
        if not temp_ds:
            raise ValueError('Nada Data. Eek.')

        # -- non-negative values
        keep_mask = np.full(temp_ds['co2'].shape, True)
        keep_mask = keep_mask & (temp_ds['co2'] > 0)
        # -- an altitude range
        keep_mask = keep_mask & (temp_ds['altitude'] <= altitude_range[1]) & (temp_ds['altitude'] > altitude_range[0])
        #
        temp_da = temp_ds['co2'].where(keep_mask)
        # temp_data = np.ma.masked_equal(temp_data, 0).data

        # Color scheme with limits at the first and last decile.
        levels = np.arange(np.floor(temp_da.quantile(0.1).values.item()),
                           np.ceil(temp_da.quantile(0.9).values.item()),
                           1).tolist()
        cmap = plt.get_cmap('magma')
        cnorm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # PLOT
        fig, ax = cls.plot_aircraft_co2_map(longitudes=temp_da['longitude'],
                                            latitudes=temp_da['latitude'],
                                            values=temp_da,
                                            cmap=cmap, cnorm=cnorm)

        return fig, ax

    def __repr__(self):
        """ String representation is built."""
        strrep = f"-- Obspack Aircraft Collection -- \n" \
                 f"Datasets:" \
                 f"\n\t" + \
                 self._original_datasets_list_str() + \
                 f"\n" \
                 f"All attributes:" \
                 f"\n\t" + \
                 '\n\t'.join(self._obj_attributes_list_str())

        return strrep
