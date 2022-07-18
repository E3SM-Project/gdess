import pytest
import pathlib

from gdess.recipes import surface_trends


@pytest.fixture
def globalview_test_data_path(rootdir: pathlib.PurePath):
    return rootdir / 'test_data' / 'globalview'


def test_recipe_input_year_error(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCC.esm-hist',
        'start_yr': "198012",
        'end_yr': "201042",
        'figure_savepath': './outputs',
        'station_list': 'mlo'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_input_model_error(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCasdasdjkhgC',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': './outputs',
        'station_list': 'mlo'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_input_stationcode_error(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCC.esm-hist',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': './outputs',
        'station_list': 'asdkjhfasg'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_completes_with_no_errors(globalview_test_data_path):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'model_name': 'BCC.esm-hist',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': './output',
        'station_list': 'mlo'}
    try:
        data_dict = surface_trends(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'surface_trends' raised an exception {exc}"
