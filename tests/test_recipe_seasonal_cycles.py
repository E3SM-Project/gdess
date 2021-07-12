import os
import pytest

from co2_diag.recipes import seasonal_cycles


def test_recipe_input_year_error(rootdir):
    test_path = os.path.join(rootdir, 'test_data')

    recipe_options = {
        'ref_data': test_path,
        'model_name': 'BCC.esm-hist',
        'station_code': 'mlo',
        'start_yr': "198012",
        'end_yr': "201042",
        'figure_savepath': 'test_figure'}
    with pytest.raises(SystemExit):
        seasonal_cycles(verbose='DEBUG', options=recipe_options)


def test_recipe_input_model_error(rootdir):
    test_path = os.path.join(rootdir, 'test_data')

    recipe_options = {
        'ref_data': test_path,
        'model_name': 'BCasdasdjkhgC',
        'station_code': 'mlo',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': 'test_figure'}
    with pytest.raises(SystemExit):
        seasonal_cycles(verbose='DEBUG', options=recipe_options)


def test_recipe_input_stationcode_error(rootdir):
    test_path = os.path.join(rootdir, 'test_data')

    recipe_options = {
        'ref_data': test_path,
        'model_name': 'BCC.esm-hist',
        'station_code': 'asdkjhfasg',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': 'test_figure'}
    with pytest.raises(SystemExit):
        seasonal_cycles(verbose='DEBUG', options=recipe_options)


def test_recipe_completes_with_no_errors(rootdir):
    test_path = os.path.join(rootdir, 'test_data')

    recipe_options = {
        'ref_data': test_path,
        'model_name': 'BCC.esm-hist',
        'station_code': 'mlo',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': 'test_figure'}
    try:
        data_dict = seasonal_cycles(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'surface_trends' raised an exception {exc}"


def test_recipe_bin_multiple_stations_completes_with_no_errors(rootdir):
    test_path = os.path.join(rootdir, 'test_data')

    recipe_options = {
        'ref_data': test_path,
        'latitude_bin_size': 30,
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': 'test_figure',
        'station_list': 'mlo'}
    try:
        data_dict = seasonal_cycles(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'surface_trends' raised an exception {exc}"
