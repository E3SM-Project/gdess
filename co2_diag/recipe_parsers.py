from co2_diag import load_stations_dict, load_config_file
from co2_diag.data_source.models.cmip.cmip_name_utils import matched_model_and_experiment, cmip_model_choices
from co2_diag.data_source.observations.gvplus_name_utils import valid_surface_stations
from co2_diag.formatters.args import valid_existing_path, valid_year_string, options_to_args, valid_writable_path
from co2_diag.operations.time import year_to_datetime64
import argparse, os, logging
from typing import Union, Callable

_logger = logging.getLogger(__name__)

stations_dict = load_stations_dict()


def add_shared_arguments_for_recipes(parser: argparse.ArgumentParser) -> None:
    """Add common recipe arguments to a parser object

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    config = load_config_file()
    default_save_path = config.get('save_path', 'value', vars=os.environ)

    parser.add_argument('ref_data', nargs='?', default=None, type=valid_existing_path,
                        help='Filepath to the reference data folder')
    parser.add_argument('--start_yr', default="1958", type=valid_year_string,
                        help='Initial year cutoff. Default is 1958, which is the first year of the Mauna Loa CO2 record.')
    parser.add_argument('--end_yr', default="2014", type=valid_year_string,
                        help='Final year cutoff. Default is 2014, which is the final year for CMIP6 historical runs.')
    parser.add_argument('--figure_savepath', default=default_save_path,
                        type=valid_writable_path, help='Filepath for saving generated figures')


def parse_recipe_options(options: Union[dict, argparse.Namespace],
                         recipe_specific_argument_adder: Callable[[argparse.ArgumentParser], None]
                         ) -> argparse.Namespace:
    """

    Parameters
    ----------
    options : Union[dict, argparse.Namespace]
        specifications for a given recipe execution
    recipe_specific_argument_adder : function
        a function that adds arguments defined for a particular recipe to a parser object

    Returns
    -------
    a parsed argument namespace
    """
    parser = argparse.ArgumentParser(description='Process surface observing station and CMIP data and compare. ')
    recipe_specific_argument_adder(parser)

    if isinstance(options, dict):
        # In this case, the options have not yet been parsed.
        params = options_to_args(options)
        if '--ref_data' in params:
            params.remove('--ref_data')  # remove this key because it is handled as a positional argument, not a kwarg.
        _logger.debug('Parameter argument string == %s', params)
        args = parser.parse_args(params)
    elif isinstance(options, argparse.Namespace):
        # In this case, the options have been parsed previously.
        args = options
    else:
        raise TypeError('<%s> is an unexpected type of the recipe options', type(options))

    _logger.debug(f"Parsed argument parameters: {args}")

    # Convert times to numpy.datetime64
    args.start_datetime = year_to_datetime64(args.start_yr)
    args.end_datetime = year_to_datetime64(args.end_yr)

    _logger.debug("Parsing is done.")
    return args


def add_surface_trends_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """Add recipe arguments to a parser object

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    add_shared_arguments_for_recipes(parser)
    parser.add_argument('--model_name', default='CMIP.NOAA-GFDL.GFDL-ESM4.esm-hist.Amon.gr1',
                        type=matched_model_and_experiment, choices=cmip_model_choices)
    parser.add_argument('--cmip_load_method', default='pangeo',
                        type=str, choices=['pangeo', 'local'])
    parser.add_argument('--difference', action='store_true')
    parser.add_argument('--globalmean', action='store_true')
    parser.add_argument('--data_savepath', default='')
    parser.add_argument('--station_list', nargs='*', type=valid_surface_stations, default=['mlo'])


def add_seasonal_cycle_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """Add recipe arguments to a parser object

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    add_shared_arguments_for_recipes(parser)
    parser.add_argument('--model_name', default='',
                        type=matched_model_and_experiment, choices=cmip_model_choices)
    parser.add_argument('--cmip_load_method', default='pangeo',
                        type=str, choices=['pangeo', 'local'])
    parser.add_argument('--difference', action='store_true')
    parser.add_argument('--latitude_bin_size', default=None, type=float)
    parser.add_argument('--plot_filter_components', action='store_true')
    parser.add_argument('--globalmean', action='store_true')
    parser.add_argument('--use_mlo_for_detrending', action='store_true')
    parser.add_argument('--run_all_stations', action='store_true')
    parser.add_argument('--data_savepath', default='')
    parser.add_argument('--station_list', nargs='*', type=valid_surface_stations, default=['mlo'])


def add_meridional_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """Add recipe arguments to a parser object

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    add_shared_arguments_for_recipes(parser)
    parser.add_argument('--model_name', default='',
                        type=matched_model_and_experiment, choices=cmip_model_choices)
    parser.add_argument('--cmip_load_method', default='pangeo',
                        type=str, choices=['pangeo', 'local'])
    parser.add_argument('--difference', action='store_true')
    parser.add_argument('--latitude_bin_size', default=None, type=float)
    parser.add_argument('--region_name', default=None, type=str,
                        help="use the same name as in the config file, e.g., 'Boreal North America'.")

    parser.add_argument('--plot_filter_components', action='store_true')
    parser.add_argument('--globalmean', action='store_true')
    parser.add_argument('--use_mlo_for_detrending', action='store_true')
    parser.add_argument('--run_all_stations', action='store_true')
    parser.add_argument('--data_savepath', default='')
    parser.add_argument('--station_list', nargs='*', type=valid_surface_stations, default=['mlo'])
