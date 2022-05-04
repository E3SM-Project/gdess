import os
import pytest

from gdess.recipes import seasonal_cycles


@pytest.fixture
def globalview_test_data_path(rootdir):
    return os.path.join(rootdir, 'test_data', 'globalview')


def test_recipe_input_year_error(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCC.esm-hist',
        'start_yr': "198012",
        'end_yr': "201042",
        'figure_savepath': './outputs',
        'station_list': 'mlo'}
    with pytest.raises(SystemExit):
        seasonal_cycles(verbose='DEBUG', options=recipe_options)


def test_recipe_input_model_error(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCasdasdjkhgC',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': './outputs',
        'station_list': 'mlo'}
    with pytest.raises(SystemExit):
        seasonal_cycles(verbose='DEBUG', options=recipe_options)


def test_recipe_input_stationcode_error(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCC.esm-hist',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': './outputs',
        'station_list': 'asdkjhfasg'}
    with pytest.raises(SystemExit):
        seasonal_cycles(verbose='DEBUG', options=recipe_options)


def test_recipe_completes_with_no_errors(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCC.esm-hist',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': './outputs',
        'station_list': 'mlo'}
    try:
        data_dict = seasonal_cycles(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'seasonal_cycles' raised an exception {exc}"


def test_recipe_bin_multiple_stations_completes_with_no_errors(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'latitude_bin_size': 30,
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': './outputs',
        'station_list': 'mlo smo'}
    try:
        data_dict = seasonal_cycles(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'seasonal_cycles' raised an exception {exc}"
