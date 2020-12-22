import glob
import numpy as np
from typing import Union

import co2_diag.dataset_operations as co2ops
from co2_diag.dataset_operations.multiset import Multiset, run_recipe
from co2_diag.dataset_operations.datasetdict import DatasetDict

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import logging
_loader_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))

station_names = {'mlo': 'Mauna Loa',
                 'brw': 'Barrow',
                 'spo': 'South Pole Observatory',
                 'smo': 'American Samoa',
                 'zep': 'Zeppelin Observatory',
                 'psa': 'Palmer Station'}


class Collection(Multiset):
    def __init__(self, verbose=False):
        """

        Parameters
        ----------
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."

        """
        self.set_verbose(verbose)

        self.df_combined_and_resampled = None
        # Define the stations that will be included in the dataset and available for diagnostic plots
        self.station_dict = {'mlo': {'name': 'Mauna Loa'},
                             'brw': {'name': 'Barrow'},
                             'spo': {'name': 'South Pole Observatory'},
                             'smo': {'name': 'American Samoa'},
                             'zep': {'name': 'Zeppelin Observatory'},
                             'psa': {'name': 'Palmer Station'}}
        _loader_logger.info("Loading data for %d observing stations..", len(self.station_dict))

        super().__init__(verbose=verbose)

    @classmethod
    @run_recipe
    def run_recipe_for_timeseries(cls,
                                  datadir='',
                                  verbose=False,
                                  param_kw: dict = None):
        """Execute a series of preprocessing steps and generate a diagnostic result.

        Parameters
        ----------
        datadir
        verbose
            can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        param_kw
            An optional dictionary with zero or more of these parameter keys:
                stationshortname (str): 'brw' is default
                start_yr (str): '1960' is default
                end_yr (str): '2015' is default

        Returns
        -------

        """
        # An instance of this Obspack Collection is created.
        new_self = cls(verbose=verbose)
        # Data are formatted into the basic data structure common to various diagnostics.
        new_self.preprocess(datadir=datadir)

        # --- Parse additional Parameters ---
        _loader_logger.debug("Parsing additional parameters ---")
        # Default values are given here.
        sc = 'brw'
        start_yr = "1960"
        end_yr = "2015"
        if param_kw:
            if 'stationshortname' in param_kw:
                if param_kw['stationshortname'] in new_self.station_dict:
                    sc = param_kw['stationshortname']
                else:
                    raise ValueError('Unexpected station name <%s>', param_kw['stationshortname'])
            if 'start_yr' in param_kw:
                start_yr = param_kw['start_yr']
            if 'end_yr' in param_kw:
                end_yr = param_kw['end_yr']

        new_self.df_combined_and_resampled = new_self.get_resampled_dataframe(new_self.stepA_original_datasets[sc],
                                                                              timestart=np.datetime64(start_yr),
                                                                              timeend=np.datetime64(end_yr)
                                                                              ).reset_index()

        # --- Plotting ---
        fig, ax = new_self.station_time_series(stationshortname=sc)

        return new_self

    def preprocess(self, datadir: str):
        """Set up the dataset that is common to every diagnostic

        Parameters
        ----------
        datadir

        Returns
        -------

        """
        _loader_logger.debug("Preprocessing ---")
        self.stepA_original_datasets = DatasetDict(self._load_stations(self.station_dict, datadir))

    @staticmethod
    def get_resampled_dataframe(dataset_obs,
                                timestart: np.datetime64,
                                timeend: np.datetime64):
        # ----------------------
        # ---- OBSERVATIONS ----
        # ----------------------
        # Time period is selected.
        ds_sub_obs = co2ops.time.select_between(dataset=dataset_obs,
                                                timestart=timestart, timeend=timeend,
                                                varlist=['time', 'value'],
                                                drop_dups=True)

        # Dataset converted to DataFrame.
        df_prepd_obs_orig = ds_sub_obs.to_dataframe().reset_index()
        df_prepd_obs_orig.rename(columns={'value': 'obs_original_resolution'}, inplace=True)

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
                               .rename(columns={'value': 'obs_resampled_resolution'})
                               )

        # ------------------
        # ---- COMBINED ----
        # ------------------
        df_prepd = (df_prepd_obs_resamp
                    .merge(df_prepd_obs_orig, on='time', how='outer')
                    .reset_index()
                    .loc[:, ['time', 'obs_original_resolution', 'obs_resampled_resolution']]
                    )

        return df_prepd

    @staticmethod
    def _load_stations(station_dict: dict,
                       datadir: str
                       ) -> dict:
        ds_obs_dict = {}
        for stationcode, _ in station_dict.items():
            _loader_logger.debug(stationcode)

            file_list = glob.glob(datadir + f"co2_{stationcode}*.nc")
            # print("files: ")
            # print(*[os.path.basename(x) for x in file_list], sep = "\n")

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
        _loader_logger.debug("Converting datetime format and units..")
        for k, v in ds_obs_dict.items():
            _loader_logger.debug(k)
            ds_obs_dict[k] = (v
                              .pipe(co2ops.time.to_datetime64)
                              .set_coords(['time', 'time_decimal', 'latitude', 'longitude', 'altitude'])
                              .sortby(['time'])
                              .swap_dims({"obs": "time"})
                              .pipe(co2ops.convert.co2_molfrac_to_ppm, co2_var_name='value')
                              )

        return ds_obs_dict

    def station_time_series(self, stationshortname: str):
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(7, 5))

        ax.plot(self.df_combined_and_resampled['time'], self.df_combined_and_resampled['obs_original_resolution'],
                label='NOAA Obs',
                marker='+', linestyle='None', color='#C0C0C0', alpha=0.6)
        ax.plot(self.df_combined_and_resampled['time'], self.df_combined_and_resampled['obs_resampled_resolution'],
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
        ax.grid(True, linestyle='--', color='gray', alpha=1)
        for spine in ax.spines.values():
            spine.set_visible(False)

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

        plt.tight_layout()
        plt.show()

        return fig, ax
        # plt.savefig('obspack_example_trend_20201123.png', dpi=500)
        # plt.savefig('obspack_example_trend_with_botlegend_20201123.png', dpi=500,
        #            bbox_extra_artists=(leg,), bbox_inches='tight')

    def __repr__(self):
        obj_attributes = sorted([k for k in self.__dict__.keys()
                                 if not k.startswith('_')])

        # String representation is built.
        strrep = f"-- Obspack Collection -- \n" \
                 f"Datasets:" \
                 f"\n\t" + \
                 self.original_datasets_list_str() + \
                 f"\n" \
                 f"All attributes:" \
                 f"\n\t" + \
                 '\n\t'.join(obj_attributes)

        return strrep

    def set_verbose(self, verbose: Union[bool, str] = False):
        # verbose can be either True, False, or a string for level such as "INFO, DEBUG, etc."
        _loader_logger.setLevel(self._validate_verbose(verbose))
