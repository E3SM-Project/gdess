import os
import pytest

from co2_diag.recipes import surface_trends


@pytest.fixture
def globalview_test_data_path(rootdir):
    return os.path.join(rootdir, 'test_data', 'globalview')


def test_recipe_input_year_error(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCC.esm-hist',
        'station_code': 'mlo',
        'start_yr': "198012",
        'end_yr': "201042",
        'figure_savepath': 'test_figure'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_input_model_error(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCasdasdjkhgC',
        'station_code': 'mlo',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': 'test_figure'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_input_stationcode_error(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCC.esm-hist',
        'station_code': 'asdkjhfasg',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': 'test_figure'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_completes_with_no_errors(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCC.esm-hist',
        'station_code': 'mlo',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': 'test_figure'}
    try:
        data_dict = surface_trends(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'surface_trends' raised an exception {exc}"
