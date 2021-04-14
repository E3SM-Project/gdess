import os
import xarray as xr
from typing import Union

from co2_diag import validate_verbose
import co2_diag.data_source as co2ops
from co2_diag.data_source.multiset import Multiset
from co2_diag.data_source.datasetdict import DatasetDict
from co2_diag.operations.time import to_datetime64
from co2_diag.operations.convert import co2_molfrac_to_ppm

import logging
_loader_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))


class ObspackCollection(Multiset):
    def __init__(self, verbose=False):
        """

        Parameters
        ----------
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        self.set_verbose(verbose)
        super().__init__(verbose=verbose)

    @staticmethod
    def _load_data_with_regex(datadir: str,
                             compiled_regex_pattern=None,
                             ) -> DatasetDict:
        """Load into memory the data from regex-defined files of Globalview+.

        Parameters
        ----------
        datadir
            directory containing the Globalview+ NetCDF files.
        compiled_regex_pattern

        Returns
        -------
        dict
            Names, latitudes, longitudes, and altitudes of each station
        """
        # --- Go through files and extract all files found via the regex pattern search ---
        file_dict = {s.group(1): f for f in os.listdir(datadir) if (s := compiled_regex_pattern.search(f))}
        # print(*[os.path.basename(x) for x in file_dict.values()], sep = "\n")

        ds_obs_dict = {}
        site_dict = {}
        for i, (sitecode, f) in enumerate(file_dict.items()):
            ds_obs_dict[sitecode] = co2ops.obspack.load.dataset_from_filelist(
                [os.path.join(datadir, f)])  # .set_index({'obs': 'obspack_num'})
            site_dict[sitecode] = {'name': ds_obs_dict[sitecode].site_name}

            lats = ds_obs_dict[sitecode]['latitude'].values
            lons = ds_obs_dict[sitecode]['longitude'].values
            # Get the latitude and longitude of each station
            #     different_station_lats = np.unique(lats)
            #     different_station_lons = np.unique(lons)
            # print(f"there are {len(different_station_lons)} different latitudes for the station: {different_station_lons}")

            # Get the average lat,lon
            meanlon = lons.mean()
            if meanlon < 0:
                meanlon = meanlon + 360
            SiteLatLon = {'lat': lats.mean(), 'lon': meanlon}
            # _loader_logger.info(str(i).rjust(2) + ". " + sitecode.ljust(12) + " - " + SiteLatLon)
            _loader_logger.info("%s. %s - %s", str(i).rjust(2), sitecode.ljust(12), SiteLatLon)
            # print(f"{SiteLatLon}")

            site_dict[sitecode]['lat'] = lats.mean()
            site_dict[sitecode]['lon'] = meanlon

        # Wrangle -- Do the things to the Obs dataset.
        _loader_logger.debug("Converting datetime format and units..")
        # Do the things to the Obs dataset.
        for k, v in ds_obs_dict.items():
            ds_obs_dict[k] = (v
                              .pipe(to_datetime64)
                              .set_coords(['time', 'time_decimal', 'latitude', 'longitude', 'altitude'])
                              .sortby(['time'])
                              #                  .swap_dims({"obs": "time"})
                              .rename({'value': 'co2'})
                              .pipe(co2_molfrac_to_ppm, co2_var_name='co2')
                              .set_index(obs=['time', 'longitude', 'latitude', 'altitude'])
                              )
        #### Concatenate all sites into one large dataset, for mapping or other combined analysis purposes
        ds_all = xr.concat(ds_obs_dict.values(), dim=('obs'))

        return DatasetDict(ds_obs_dict)

    def set_verbose(self, verbose: Union[bool, str] = False) -> None:
        """
        Parameters
        ----------
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        _loader_logger.setLevel(validate_verbose(verbose))
