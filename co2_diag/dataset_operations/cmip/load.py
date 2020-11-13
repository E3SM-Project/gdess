import numpy as np
import xarray as xr

import co2_diag.dataset_operations as co2ops

# Packages for using NCAR's intake
import intake
import intake_esm

import logging
# _logger = logging.getLogger(__name__)

_loader_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))
class Loader():
    def __init__(self, verbose=False):
        url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
        self.dataframe = intake.open_esm_datastore(url)
        self.latest_searched_models = None
        self.loaded_datasets = None
        self.datasets_prepped_for_execution = {}
        if verbose:
            _loader_logger.setLevel(logging.DEBUG)

    @staticmethod
    def set_verbose(switch):
        if switch == 'on':
            _loader_logger.setLevel(logging.DEBUG)
        elif switch == 'off':
            _loader_logger.setLevel(logging.WARN)
        else:
            raise ValueError("Unexpect/unhandled verbose option <%s>. Please use 'on' or 'off'", switch)

    def search(self, **query) -> intake_esm.core.esm_datastore:
        """Wrapper for intake's catalog search

        query keyword arguments:
            experiment_id
            table_id
            variable_id
            institution_id
            member_id
            grid_label

        Returns
        -------

        """
        _loader_logger.info("query dictionary: %s", query)
        self.latest_searched_models = self.dataframe.search(**query)
        return self.latest_searched_models

    def load_datasets_from_searched_models(self):
        self.loaded_datasets = self.latest_searched_models.to_dataset_dict()
        self.datasets_prepped_for_execution = self.loaded_datasets
        _loader_logger.info('\n'.join(self.loaded_datasets.keys()))

    def apply_function_to_all_datasets(self, fnc, *args, **kwargs):
        count = 0
        for k in self.datasets_prepped_for_execution.keys():
            count += 1
            _loader_logger.debug("-- %d - {%s}.. ", count, k)
            self.datasets_prepped_for_execution[k] = self.datasets_prepped_for_execution[k].pipe(fnc, *args, **kwargs)

        if count == 0:
            _loader_logger.debug("Nothing done. No datasets are ready for execution.")
        else:
            _loader_logger.debug("all processed.")

    def load_all(self):
        self.apply_function_to_all_datasets(xr.Dataset.load)
        _loader_logger.info("done.")

    def convert_all_to_ppm(self):
        # Convert CO2 units to ppm
        self.apply_function_to_all_datasets(co2ops.convert.co2_molfrac_to_ppm, co2_var_name='co2')
        _loader_logger.info("all converted.")

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
        _loader_logger.info("all selections applied, but not yet executed. Ready for .load()")

    def apply_mean(self, dim):
        self.apply_function_to_all_datasets(xr.Dataset.mean, dim=dim)
        _loader_logger.info("mean applied to all, but not yet executed. Ready for .load()")



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

