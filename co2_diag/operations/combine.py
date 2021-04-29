# NOTE: This module is not currently being used and its contents have been completely commented-out.

# import numpy as np
# import pandas as pd
# import xarray as xr
#
# from co2_diag.operations.time import select_between
#
# def get_combined_dataframe(dataset_ref: xr.Dataset,
#                            dataset_e3sm: xr.Dataset,
#                            timestart: np.datetime64,
#                            timeend: np.datetime64,
#                            ref_var: str = 'value',
#                            e3sm_var: str = 'CO2'
#                            ) -> pd.DataFrame:
#     """Combine E3SM station data and a reference dataset
#     into a pandas Dataframe for a specified time period
#
#     Parameters
#     ----------
#     dataset_ref
#     dataset_e3sm
#     timestart
#     timeend
#     ref_var
#     e3sm_var
#
#     Returns
#     -------
#
#     """
#     # ----------------------
#     # ----- REFERENCE ------
#     # ----------------------
#     ds_sub_ref = select_between(dataset=dataset_ref,
#                                 timestart=timestart,
#                                 timeend=timeend,
#                                 varlist=['time', ref_var])
#     new_ref_varname = 'ref_' + ref_var
#     df_prepd_ref = (ds_sub_ref
#                     .to_dataframe()
#                     .reset_index()
#                     .rename(columns={ref_var: new_ref_varname})
#                     )
#
#     # # Get time-resampled resolution
#     # ds_prepd_ref_resamp = ds_sub_ref.where(tempmask, drop=True).copy()
#     # #     ds_prepd_ref_resamp = ds_prepd_ref_resamp.resample(time="1D").interpolate("linear")  # weekly average
#     # ds_prepd_ref_resamp = ds_prepd_ref_resamp.resample(time="1MS").mean()  # monthly average
#     # # ds_prepd = ds_sub.resample(time="1AS").mean()  # yearly average
#     # # ds_prepd = ds_sub.resample(time="Q").mean()  # quarterly average (consecutive three-month periods)
#     # # ds_prepd = ds_sub.resample(time="QS-DEC").mean()  # quarterly average (consecutive three-month periods), anchored at December 1st.
#     # #
#     # df_prepd_ref_resamp = (ds_prepd_ref_resamp
#     #                        .dropna(dim=('time'))
#     #                        .to_dataframe()
#     #                        .reset_index()
#     #                        .rename(columns={ref_var: 'ref_' + ref_var + '_resampled_resolution'})
#     #                        )
#
#     # ----------------------
#     # ---- E3SM Output -----
#     # ----------------------
#     ds_sub_e3sm = select_between(dataset=dataset_e3sm,
#                                  timestart=timestart,
#                                  timeend=timeend,
#                                  varlist=['time', e3sm_var])
#     new_e3sm_varname = 'e3sm_' + e3sm_var
#     df_prepd_e3sm = (ds_sub_e3sm
#                      .to_dataframe()
#                      .reset_index()
#                      .drop(columns=['ncol', 'lat', 'lon'])
#                      .rename(columns={e3sm_var: new_e3sm_varname})
#                      )
#     # get the vertical mean
#     #     ds_prepd_e3sm_orig = ds_prepd_e3sm_orig.mean(dim='lev').where(ds_mdl['ncol']==closest_mdl_point_dict['index'], drop=True)
#
#     # get the lowest level
#     ds_prepd_e3sm_orig = (ds_prepd_e3sm_orig
#                           .sel(lev=dataset_e3sm['lev'][-1])
#                           .where(dataset_e3sm['ncol'] == closest_mdl_point_dict['index'], drop=True)
#                           )
#
#     # Resample the time
#     #     ds_prepd_e3sm_resamp = df_prepd_e3sm_orig.resample(time="1D").interpolate("linear")  # weekly average
#
#     # ------------------
#     # ---- COMBINED ----
#     # ------------------
#     df_prepd = (df_prepd_ref
#                 .merge(df_prepd_e3sm, on='time', how='outer')
#                 .reset_index()
#                 .loc[:, ['time', new_ref_varname, new_e3sm_varname]]
#                 )
#     # df_prepd['obs_original_resolution'] = df_prepd['obs_original_resolution'].astype(float)
#     # df_prepd['obs_resampled_resolution'] = df_prepd['obs_resampled_resolution'].astype(float)
#     # df_prepd['model_original_resolution'] = df_prepd['model_original_resolution'].astype(float)
#     # df_prepd.rename(columns={'obs_original_resolution': 'NOAA Obs',
#     #                          'obs_resampled_resolution': 'NOAA Obs daily mean',
#     #                          'model_original_resolution': 'E3SM'}, inplace=True)
#
#     return df_prepd
