import xarray as xr
from dask.diagnostics import ProgressBar

import logging

_multiset_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))
class Multiset():
    """Useful class for working simultaneously with multiple, consistent xarray Datasets

    """
    def __init__(self, verbose=False):
        """

        Parameters
        ----------
        verbose
        """
        self.original_datasets = None
        self.datasets_prepped_for_execution = {}
        self.latest_executed_datasets = {}
        if verbose:
            _multiset_logger.setLevel(logging.DEBUG)

    def __repr__(self):
        obj_attributes = sorted([k for k in self.__dict__.keys()
                                 if not k.startswith('_')])

        # String representation is built.
        strrep = f"Multiset: \n" + \
                 '\t\n'.join(self.original_datasets.keys()) + \
                 f"\n" \
                 f"\t all attributes:%s" % '\n\t\t\t'.join(obj_attributes)

        return strrep

    @staticmethod
    def set_verbose(switch):
        if switch == 'on':
            _multiset_logger.setLevel(logging.DEBUG)
        elif switch == 'off':
            _multiset_logger.setLevel(logging.WARN)
        else:
            raise ValueError("Unexpect/unhandled verbose option <%s>. Please use 'on' or 'off'", switch)

    def apply_function_to_all_datasets(self, fnc, *args, **kwargs):
        """

        Parameters
        ----------
        fnc
        args
        kwargs:
            'executing' (bool): specify whether we are executing calculations or lazily applying a new calculation

        Returns
        -------

        """

        destination_dict = self.datasets_prepped_for_execution
        if 'executing' in kwargs:
            if kwargs['executing']:
                destination_dict = self.latest_executed_datasets

        count = 0
        for k in self.datasets_prepped_for_execution.keys():
            count += 1
            _multiset_logger.debug("-- %d - {%s}.. ", count, k)
            destination_dict[k] = self.datasets_prepped_for_execution[k].pipe(fnc, *args, **kwargs)

        if count == 0:
            _multiset_logger.debug("Nothing done. No datasets are ready for execution.")
        else:
            _multiset_logger.debug("all processed.")

    def load_all(self, progressbar=True):
        if progressbar:
            ProgressBar().register()

        self.apply_function_to_all_datasets(xr.Dataset.load, executing=True)
        _multiset_logger.info("done.")

    def apply_selection(self, **selection_dict):
        """Select from dataset.  Wrapper for Xarray's .sel().

        Example
        -------
        One can pass slices or individual values:
            cmip_obj.apply_selection(time=slice("1960", None))
            cmip_obj.apply_selection(plev=100000)

        Parameters
        ----------
        selection_dict

        Returns
        -------

        """
        self.apply_function_to_all_datasets(xr.Dataset.sel, **selection_dict)
        _multiset_logger.info("all selections applied, but not yet executed. Ready for .load()")

    def apply_mean(self, dim):
        self.apply_function_to_all_datasets(xr.Dataset.mean, dim=dim)
        _multiset_logger.info("mean applied to all, but not yet executed. Ready for .load()")

    # def lineplots(self):
    #     # plt.rcParams.update({'font.size': 12,
    #     #                      'lines.linewidth': 2,
    #     #                      })
    #
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    #
    #     for ki, k in enumerate(self.latest_executed_datasets.keys()):
    #         for mi, m in enumerate(self.latest_executed_datasets[k]['member_id'].values.tolist()):
    #             color_count = ki * max(member_counts) + mi
    #
    #             darray = dataset_dict[k].sel(member_id=m)
    #
    #             # Some time variables are numpy datetime64, some are CFtime.  Errors are raised if plotted together.
    #             if isinstance(darray['time'].values[0], np.datetime64):
    #                 pass
    #             else:
    #                 # Warnings are raised when converting CFtimes to datetimes, because subtle errors.
    #                 with warnings.catch_warnings():
    #                     warnings.simplefilter("ignore")
    #                     darray = co2ops.time.to_datetimeindex(darray)
    #
    #             ax.plot(darray['time'], darray.to_array().squeeze(), label=f"{k} ({m})",
    #                     color=my_cmap.colors[color_count], alpha=0.6)
    #
    #     ax.set_ylabel('ppm')
    #     ax.grid(True, linestyle='--', color='gray', alpha=1)
    #     for spine in plt.gca().spines.values():
    #         spine.set_visible(False)
    #
    #     leg = plt.legend(title='Models', frameon=False,
    #                      bbox_to_anchor=(1.05, 1), loc='upper left',
    #                      fontsize=12)
    #
    #     plt.tight_layout()
    #     plt.show()



# def datasets(file_list: list,
#                           vars_to_keep: list = None,
#                           decode_times: bool = False):
#     """Load ObsPack NetCDF files specified in a list and create one Dataset from them.
#
#     :param file_list:
#     :param vars_to_keep: list
#     :param decode_times: parameter passed to Xarray.open_dataset()
#     :return: xr.Dataset
#     """
#     if vars_to_keep is None:
#         # These are the default variables to keep if not overridden by a passed parameter.
#         vars_to_keep = ['value', 'nvalue', 'value_std_dev',
#                         'time', 'start_time', 'datetime', 'time_decimal',
#                         'latitude', 'longitude', 'altitude',
#                         'qcflag', 'dataset_platform', 'dataset_project',
#                         'obspack_num', 'obspack_id']
#
#     ds_list = []
#     for i, f in enumerate(file_list):
#         thisds = xr.open_dataset(f, decode_times=decode_times)
#
#         # If the following variables are not present, continue loading and just make them blank DataArrays
#         #    Otherwise, we will raise an error
#         possible_missing_vars = ['qcflag', 'value_std_dev', 'nvalue']
#         for pmv in possible_missing_vars:
#             if not (pmv in thisds.keys()):
#                 blankarray = xr.DataArray(data=[np.nan], dims='obs', name=pmv).squeeze()
#                 thisds = thisds.assign({pmv: blankarray})
#
#         # Only the specified variables are retained.
#         to_drop = []
#         for vname in thisds.keys():
#             if not (vname in vars_to_keep):
#                 to_drop.append(vname)
#         newds = thisds.drop_vars(to_drop)
#
#         # Dataset attributes 'platform' and 'project' are copied to every data point along the 'obs' dimension.
#         n_obs = len(thisds['obs'])
#         newds = newds.assign(dataset_platform=xr.DataArray([thisds.attrs['dataset_platform']] * n_obs, dims='obs'))
#         newds = newds.assign(dataset_project=xr.DataArray([thisds.attrs['dataset_project']] * n_obs, dims='obs'))
#
#         ds_list.append(newds)
#         #     if i > 100:
#         #         break
#
#     ds = xr.concat(ds_list, dim='obs')
#
#     return ds


def intake_wrapper(datastore='cmip6',
                   **query):
    """

    Parameters
    ----------
    datastore
    query:
        experiment_id
        table_id
        variable_id
        institution_id
        member_id
        grid_label

    Returns
    -------

    """

    # if datastore == 'cmip6':
    #     url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
    #     dataframe = intake.open_esm_datastore(url)
    # else:
    #     raise ValueError('Unexpected/unhandled datastore <%s>', datastore)
    #
    # models = dataframe.search(**query)
    #
    # return dataframe, models

