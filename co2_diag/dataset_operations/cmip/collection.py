import numpy as np
import warnings

import co2_diag.dataset_operations as co2ops
from co2_diag.dataset_operations.multiset import Multiset

# Packages for using NCAR's intake
import intake
import intake_esm

import matplotlib as mpl
import matplotlib.pyplot as plt

import logging

_loader_logger = logging.getLogger("{0}.{1}".format(__name__, "loader"))
class Collection(Multiset):
    def __init__(self, datastore='cmip6', verbose=False):
        """

        Parameters
        ----------
        datastore
        verbose
        """
        if datastore == 'cmip6':
            url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
        else:
            raise ValueError('Unexpected/unhandled datastore <%s>', datastore)

        self.dataframe = intake.open_esm_datastore(url)
        self.latest_searched_models = None

        self.original_datasets = None
        self.datasets_prepped_for_execution = {}
        self.latest_executed_datasets = {}
        super(Multiset, self).__init__()

        if verbose:
            _loader_logger.setLevel(logging.DEBUG)
        else:
            _loader_logger.setLevel(logging.WARN)

    def count_members(self, verbose=True):
        # Get the number of member_id values present for each model's dataset.
        member_counts = []
        for k in self.original_datasets.keys():
            member_counts.append(len(self.original_datasets[k]['member_id'].values))
        nmodels = len(member_counts)
        if verbose:
            _loader_logger.info(f"There are <%s> members for each of the %d models.", member_counts, nmodels)

        return nmodels, member_counts

    def __repr__(self):
        obj_attributes = sorted([k for k in self.__dict__.keys()
                                 if not k.startswith('_')])

        nmodels, member_counts = self.count_members(verbose=False)

        # String representation is built.
        strrep = f"-- CMIP Loader -- \n" \
                 f"Datasets:" \
                 f"\n\t" + \
                 '\n\t'.join(self.original_datasets.keys()) + \
                 f"\n" + \
                 f"There are <{member_counts}> members for each of the {nmodels} models." \
                 f"\n" \
                 f"All attributes:" \
                 f"\n\t" + \
                 '\n\t'.join(obj_attributes)

        return strrep

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
        self.original_datasets = self.latest_searched_models.to_dataset_dict()
        self.datasets_prepped_for_execution = self.original_datasets
        _loader_logger.info('\n'.join(self.original_datasets.keys()))

        _loader_logger.info("Converting units to ppm.")
        self.convert_all_to_ppm()

    def convert_all_to_ppm(self):
        # Convert CO2 units to ppm
        self.apply_function_to_all_datasets(co2ops.convert.co2_molfrac_to_ppm, co2_var_name='co2')
        _loader_logger.info("all converted.")

    @staticmethod
    def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
        """
        Params:
            nc: number of categories
            nsc: number of subcategories
            cmap:
            continuous:

        Returns:
            A colormap with nc*nsc different colors, where for each category there are nsc colors of same hue

        Notes:
            from https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
        """
        if nc > plt.get_cmap(cmap).N:
            raise ValueError("Too many categories for colormap.")
        if continuous:
            ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
        else:
            ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
        cols = np.zeros((nc * nsc, 3))
        for i, c in enumerate(ccolors):
            chsv = mpl.colors.rgb_to_hsv(c[:3])
            arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
            arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
            arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
            rgb = mpl.colors.hsv_to_rgb(arhsv)
            cols[i * nsc:(i + 1) * nsc, :] = rgb
        cmap = mpl.colors.ListedColormap(cols)
        return cmap

    def lineplots(self):
        # plt.rcParams.update({'font.size': 12,
        #                      'lines.linewidth': 2,
        #                      })

        nmodels, member_counts = self.count_members()
        my_cmap = self.categorical_cmap(nc=len(member_counts), nsc=max(member_counts),
                                        cmap="tab10")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))

        for ki, k in enumerate(self.latest_executed_datasets.keys()):
            for mi, m in enumerate(self.latest_executed_datasets[k]['member_id'].values.tolist()):
                color_count = ki * max(member_counts) + mi

                darray = self.latest_executed_datasets[k].sel(member_id=m)

                # Some time variables are numpy datetime64, some are CFtime.  Errors are raised if plotted together.
                if isinstance(darray['time'].values[0], np.datetime64):
                    pass
                else:
                    # Warnings are raised when converting CFtimes to datetimes, because subtle errors.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        darray = co2ops.time.to_datetimeindex(darray)

                ax.plot(darray['time'], darray.to_array().squeeze(), label=f"{k} ({m})",
                        color=my_cmap.colors[color_count], alpha=0.6)

        ax.set_ylabel('ppm')
        ax.grid(True, linestyle='--', color='gray', alpha=1)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        leg = plt.legend(title='Models', frameon=False,
                         bbox_to_anchor=(1.05, 1), loc='upper left',
                         fontsize=12)

        plt.tight_layout()
        plt.show()
