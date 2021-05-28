import os
import re
import glob
import argparse
from typing import Union

import numpy as np
import pandas as pd

from co2_diag import set_verbose
import co2_diag.data_source as co2ops
from co2_diag.data_source.obspack.load import load_data_with_regex
from co2_diag.data_source.multiset import Multiset
from co2_diag.data_source.datasetdict import DatasetDict

from co2_diag.operations.time import select_between, ensure_dataset_datetime64, \
    ensure_datetime64_array, year_to_datetime64
from co2_diag.operations.convert import co2_molfrac_to_ppm

from co2_diag.graphics.utils import aesthetic_grid_no_spines, mysavefig
from co2_diag.recipes.utils import benchmark_recipe, add_shared_arguments_for_recipes, parse_recipe_options

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import logging
_loader_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))

# Define the stations that will be included in the dataset and available for diagnostic plots
station_dict = {'mlo': {'name': 'Mauna Loa'},
                'brw': {'name': 'Barrow'},
                'spo': {'name': 'South Pole Observatory'},
                'smo': {'name': 'American Samoa'},
                'zep': {'name': 'Zeppelin Observatory'},
                'psa': {'name': 'Palmer Station'}}


class Collection(Multiset):
    def __init__(self, verbose: Union[bool, str]=False):
        """Instantiate an Obspack Surface Station Collection object.

        Parameters
        ----------
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        """
        set_verbose(_loader_logger, verbose)

        self.df_combined_and_resampled = None
        # Define the stations that will be included in the dataset and available for diagnostic plots
        self.station_dict = station_dict.copy()

        super().__init__(verbose=verbose)

    @classmethod
    @benchmark_recipe
    def run_recipe_for_timeseries(cls,
                                  verbose: Union[bool, str] = False,
                                  options: dict = None
                                  ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        verbose: Union[bool, str]
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        options
            A dictionary with zero or more of these parameter keys:
                ref_data (str): directory containing the NOAA Obspack NetCDF files
                station_code (str): 'mlo' is default
                start_yr (str): '1960' is default
                end_yr (str): '2015' is default

        Returns
        -------
        Collection object for Obspack that was used to generate the diagnostic
        """
        set_verbose(_loader_logger, verbose)
        opts = parse_recipe_options(options, add_surface_station_collection_args_to_parser)

        # An empty instance is created.
        new_self = cls(verbose=verbose)

        # --- Apply diagnostic parameters and prep data for plotting ---
        # Data are formatted into the basic data structure common to various diagnostics.
        new_self.preprocess(datadir=opts.ref_data, station_name=opts.station_code)
        # Data are resampled
        new_self.df_combined_and_resampled = (new_self
                                              .get_resampled_dataframe(new_self.stepA_original_datasets[opts.station_code],
                                                                       timestart=opts.start_datetime,
                                                                       timeend=opts.end_datetime
                                                                       ).reset_index())

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_station_time_series(stationshortname=opts.station_code)
        if opts.figure_savepath:
            mysavefig(fig, opts.figure_savepath, 'cmip_timeseries', bbox_extra_artists=bbox_artists)

        return new_self

    @classmethod
    @benchmark_recipe
    def run_recipe_for_annual_series(cls,
                                     verbose: Union[bool, str] = False,
                                     options: dict = None
                                     ) -> 'Collection':
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        options
            A dictionary with zero or more of these parameter keys:
                ref_data (str): directory containing the NOAA Obspack NetCDF files
                start_yr (str): '1960' s default
                end_yr (str): None is default

        Returns
        -------
        Collection object for Obspack that was used to generate the diagnostic
        """
        set_verbose(_loader_logger, verbose)
        opts = parse_recipe_options(options, add_surface_station_collection_args_to_parser)

        # An empty instance is created.
        new_self = cls(verbose=verbose)

        # --- Apply diagnostic parameters and prep data for plotting ---
        # Data are formatted into the basic data structure common to various diagnostics.
        new_self.preprocess(datadir=opts.ref_data)

        _loader_logger.info('Applying selected bounds..')
        # Data are resampled
        new_self.df_combined_and_resampled = (new_self
                                              .get_resampled_dataframe(new_self.stepA_original_datasets[opts.station_code],
                                                                       timestart=opts.start_datetime,
                                                                       timeend=opts.end_datetime
                                                                       ).reset_index())

        df_anomaly_mean_cycle, df_anomaly_yearly = (Multiset
                                                    .get_anomaly_dataframes(new_self.stepA_original_datasets[opts.station_code],
                                                                            varname='co2'))

        # --- Plotting ---
        fig, ax, bbox_artists = new_self.plot_annual_series(df_anomaly_yearly, df_anomaly_mean_cycle,
                                                            stationname=opts.station_code)
        if opts.figure_savepath:
            mysavefig(fig, opts.figure_savepath, 'obspack_annual_series', bbox_extra_artists=bbox_artists)

        return new_self

    def preprocess(self, datadir: str,
                   station_name: Union[str, list] = None
                   ) -> None:
        """Set up the dataset that is common to every diagnostic

        Parameters
        ----------
        datadir
        station_name
        """
        _loader_logger.debug("Preprocessing...")
        if not station_name:
            # Use predefined dictionary of stations at the top of this module
            stations = self.station_dict
        else:
            # Create a subset of the station dictionary containing only the station name(s) passed in
            if isinstance(station_name, str):
                station_name = [station_name]
            stations = dict((k, self.station_dict[k]) for k in station_name)

        self.stepA_original_datasets = DatasetDict(self._load_stations_by_namedict(stations, datadir))
        _loader_logger.debug("Preprocessing is done.")

    @staticmethod
    def get_resampled_dataframe(dataset_obs,
                                timestart,
                                timeend) -> pd.DataFrame:
        """Get data resampled at monthly intervals

        Parameters
        ----------
        dataset_obs
        timestart
        timeend

        Returns
        -------
        A pandas.DataFrame with columnds of time, original data, and resampled data
        """
        _loader_logger.debug('Resampling obspack observations..')
        # --- OBSERVATIONS ---
        # Time period is selected.
        ds_sub_obs = select_between(dataset=dataset_obs,
                                    timestart=timestart, timeend=timeend,
                                    varlist=['time', 'co2'],
                                    drop_dups=True)
        # Dataset converted to DataFrame.
        df_prepd_obs_orig = ds_sub_obs.to_dataframe().reset_index()
        df_prepd_obs_orig.rename(columns={'co2': 'obs_original_resolution'}, inplace=True)

        # --- Resampled observations ---
        #     ds_resampled = ds_sub_obs.resample(time="1D").interpolate("linear")  # weekly average
        ds_resampled = ds_sub_obs.resample(time="1MS").mean()  # monthly average
        # ds_resampled = ds_sub_obs.resample(time="1AS").mean()  # yearly average
        # ds_resampled = ds_sub_obs.resample(time="Q").mean()  # quarterly average (consecutive three-month periods)
        # ds_resampled = ds_sub_obs.resample(time="QS-DEC").mean()  # quarterly average (consecutive three-month periods), anchored at December 1st.
        #
        # Dataset converted to DataFrame.
        df_prepd_obs_resamp = (ds_resampled
                               .dropna(dim=('time'))
                               .to_dataframe().reset_index()
                               .rename(columns={'co2': 'obs_resampled_resolution'})
                               )

        # --- COMBINED ---
        df_prepd = (df_prepd_obs_resamp
                    .merge(df_prepd_obs_orig, on='time', how='outer')
                    .reset_index()
                    .loc[:, ['time', 'obs_original_resolution', 'obs_resampled_resolution']]
                    )

        _loader_logger.debug('  First resampled row: %s', df_prepd.iloc[0, :])
        _loader_logger.debug('Done.')

        return df_prepd

    @staticmethod
    def _load_surface_data(datadir: str,
                           ) -> DatasetDict:
        """Load into memory the data for surface measurements from Globalview+.

        Parameters
        ----------
        datadir
            directory containing the Globalview+ NetCDF files.

        Returns
        -------
        dict
            Names, latitudes, longitudes, and altitudes of each station
        """
        # --- Go through files and extract all 'surface' sampled files ---
        p = re.compile(r'co2_([a-zA-Z0-9]*)_surface.*\.nc$')
        return_value = load_data_with_regex(datadir=datadir, compiled_regex_pattern=p)
        return return_value

    @staticmethod
    def _load_stations_by_namedict(station_dict: dict,
                                   datadir: str
                                   ) -> dict:
        """Load into memory the data for surface observing stations from Globalview+.

        Parameters
        ----------
        station_dict
        datadir
            directory containing the Globalview+ NetCDF files.

        Returns
        -------
        dict
            Names, latitudes, longitudes, and altitudes of each station
        """
        ds_obs_dict = {}
        for stationcode, _ in station_dict.items():
            _loader_logger.debug(stationcode)
            _loader_logger.debug('data directory: %s', datadir)

            file_list = glob.glob(os.path.join(datadir, f"co2_{stationcode}*.nc"))
            # print("files: ")
            # print(*[os.path.basename(x) for x in file_list], sep = "\n")

            _loader_logger.debug('Station files: %s', ', '.join([os.path.basename(x) for x in file_list]))
            ds_obs_dict[stationcode] = co2ops.obspack.load.dataset_from_filelist(file_list)

            # Simple unit check - for the Altitude variable
            check_altitude_unit = ds_obs_dict[stationcode]['altitude'].attrs['units'] == 'm'
            if not check_altitude_unit:
                raise ValueError('unexpected altitude units <%s>', ds_obs_dict[stationcode]['altitude'].attrs['units'])

            lats = ds_obs_dict[stationcode]['latitude'].values
            lons = ds_obs_dict[stationcode]['longitude'].values
            alts = ds_obs_dict[stationcode]['altitude'].values

            # Get the latitude and longitude of each station
            #     different_station_lats = np.unique(lats)
            #     different_station_lons = np.unique(lons)
            # print(f"there are {len(different_station_lons)} different latitudes for the station: {different_station_lons}")

            # Get the average lat,lon
            meanlon = lons.mean()
            if meanlon < 0:
                meanlon = meanlon + 360
            station_latlonalt = {'lat': lats.mean(), 'lon': meanlon, 'alts': alts.mean()}
            _loader_logger.debug("  %s" % station_latlonalt)

            station_dict[stationcode].update(station_latlonalt)

        # Wrangle -- Do the things to the Obs dataset.
        _loader_logger.debug("Converting datetime format and units...")
        for i, (k, v) in enumerate(ds_obs_dict.items()):
            _loader_logger.debug('  %s', k)
            ds_obs_dict[k] = (v
                              .set_coords(['time', 'time_decimal', 'latitude', 'longitude', 'altitude'])
                              .sortby(['time'])
                              .swap_dims({"obs": "time"})
                              .pipe(ensure_dataset_datetime64)
                              .rename({'value': 'co2'})
                              .pipe(co2_molfrac_to_ppm, co2_var_name='co2')
                              )
            if i == 0:
                _loader_logger.debug("  the first DataSet has a time range of <%s> to <%s>.",
                                     np.datetime_as_string(ds_obs_dict[k]['time'].values[0], unit='D'),
                                     np.datetime_as_string(ds_obs_dict[k]['time'].values[-1], unit='D'))
        _loader_logger.debug("Converting is done.")

        return ds_obs_dict

    def plot_station_time_series(self, stationshortname: str) -> (plt.Figure, plt.Axes, tuple):
        """Make timeseries plot of co2 concentration for each surface observing station.

        Returns
        -------
        matplotlib figure
        matplotlib axis
        tuple
            Extra matplotlib artists used for the bounding box (bbox) when saving a figure
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(7, 5))
        ax.plot(ensure_datetime64_array(self.df_combined_and_resampled['time']),
                self.df_combined_and_resampled['obs_original_resolution'],
                label='NOAA Obs',
                marker='+', linestyle='None', color='#C0C0C0', alpha=0.6)
        ax.plot(ensure_datetime64_array(self.df_combined_and_resampled['time']),
                self.df_combined_and_resampled['obs_resampled_resolution'],
                label='NOAA Obs monthly mean',
                linestyle='-', color=(0 / 255, 133 / 255, 202 / 255), linewidth=2)
        #
        ax.set_ylim((288.5231369018555, 429.76668853759764))
        #
        # ax[i].set_ylabel('$ppm$')
        #     ax.legend(bbox_to_anchor=(1.05, 1))
        ax.set_ylabel('$CO_2$ (ppm)')
        ax.text(0.02, 0.88, f"{stationshortname.upper()}\n{self.station_dict[stationshortname]['lat']:.1f}, "
                            f"{self.station_dict[stationshortname]['lon']:.1f}",
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=16)
        #
        aesthetic_grid_no_spines(ax)

        # Define the date format
        #             ax.xaxis.set_major_locator(mdates.YearLocator())
        #             date_form = DateFormatter("%b\n%Y")
        date_form = DateFormatter("%Y")
        ax.xaxis.set_major_formatter(date_form)
        #         ax.xaxis.set_minor_locator(mdates.MonthLocator())
        #         ax.tick_params(which="both", bottom=True)

        # leg = ax.legend(loc='lower right', fontsize=14)
        leg = plt.legend(title='', frameon=False,
                         bbox_to_anchor=(0, -0.1), loc='upper left',
                         fontsize=12)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh._legmarker.set_alpha(1)
        bbox_artists = (leg,)

        return fig, ax, bbox_artists

    def plot_annual_series(self, df_anomaly_yearly, df_anomaly_cycle, stationname: str) -> (plt.Figure, plt.Axes, tuple):
        """Make timeseries plot with annual anomalies of co2 concentration.

        Returns
        -------
        matplotlib figure
        matplotlib axis
        tuple
            Extra matplotlib artists used for the bounding box (bbox) when saving a figure
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

        # ---- Plot Observations ----
        ax.plot(df_anomaly_yearly, label='annual cycle',
                color='#C0C0C0', linestyle='-', alpha=0.3, marker='.', zorder=-32)
        ax.plot(df_anomaly_cycle['moy'], df_anomaly_cycle['monthly_anomaly_from_year'],
                label='mean annual cycle', marker='o', zorder=10,
                color=(0 / 255, 133 / 255, 202 / 255))  # (255/255, 127/255, 14/255))
        #
        ax.set_ylim((-13, 7))
        #
        ax.set_ylabel('$CO_2$ (ppm)')
        ax.set_xlabel('month')
        # ax.set_title(titlestr, fontsize=12)
        #
        ax.text(0.02, 0.92, f"{stationname.upper()}, "
                            f"{self.station_dict[stationname]['lat']:.1f}, {self.station_dict[stationname]['lon']:.1f}",
                horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
        #
        # Define the legend
        handles, labels = ax.get_legend_handles_labels()
        display = (0, len(handles) - 1)
        leg = ax.legend([handle for i, handle in enumerate(handles) if i in display],
                        [label for i, label in enumerate(labels) if i in display],
                        loc='best', fontsize=12)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh._legmarker.set_alpha(1)
        #
        #         ax.grid(linestyle='--', color='lightgray')
        #         for k in ax.spines.keys():
        #             ax.spines[k].set_alpha(0.5)
        bbox_artists = (leg,)

        return fig, ax, bbox_artists

    def __repr__(self):
        """ String representation is built."""
        strrep = f"-- Obspack Surface Station Collection -- \n" \
                 f"Datasets:" \
                 f"\n\t" + \
                 self._original_datasets_list_str() + \
                 f"\n" \
                 f"All attributes:" \
                 f"\n\t" + \
                 '\n\t'.join(self._obj_attributes_list_str())

        return strrep


def get_dict_of_all_stations(datadir):
    """Build a dictionary that contains a key for each station code,
       and with a list of filenames for each key.

    Parameters
    ----------
    datadir : str
        the directory containing netcdf files for the station data

    Returns
    -------
    A dictionary with (keys) three-letter station codes, and for each station (values) a list of data filenames
    """
    filepath_list = glob.glob(datadir + '*surface*.nc')
    filenames = [os.path.basename(x) for x in filepath_list]

    # regex to get the station code from each filename
    pattern = r"co2_(?P<station_code>.*)_surface.*"

    dict_to_build = dict()
    for f in filenames:
        result = re.match(pattern, f)['station_code']
        if result not in dict_to_build.keys():
            dict_to_build[result] = [f]
        else:
            dict_to_build[result].append(f)

    return dict_to_build


def add_surface_station_collection_args_to_parser(parser: argparse.PARSER) -> None:
    """Add recipe arguments to a parser object

    Parameters
    ----------
    parser
    """
    add_shared_arguments_for_recipes(parser)
    parser.add_argument('--station_code', default='mlo',
                        type=str, choices=station_dict.keys())
