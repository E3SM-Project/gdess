from co2_diag import load_stations_dict
import xarray as xr
import glob, os, re, argparse, shlex

# -- Define valid surface station choices --
station_dict = load_stations_dict()
station_code_choices = list(station_dict.keys())


def valid_surface_stations(station_arg: str) -> str:
    """Validate that a string containing one or more station codes are present in the available dataset

    Returns
    -------
    a space-delimited string of surface station codes.
    """
    my_splitter = shlex.shlex(station_arg, posix=True)
    my_splitter.whitespace += ','
    my_splitter.whitespace_split = True
    my_list = list(my_splitter)

    for i, s in enumerate(my_list):
        if not (s in station_code_choices):
            raise argparse.ArgumentTypeError('Station name must be available in the dataset. <%s> is not.' % s)

    return station_arg


def get_dict_of_all_station_filenames(datadir):
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


def get_dict_of_station_codes_and_names(datadir):
    stations_dict = get_dict_of_all_station_filenames(datadir)
    return {k: {'name': xr.open_dataset(os.path.join(datadir, stations_dict[k][0])).attrs['site_name']}
            for k, v
            in stations_dict.items()}