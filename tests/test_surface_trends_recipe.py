import os
import pytest

from co2_diag.data_source.obspack.surface_stations.collection import Collection
from co2_diag.recipes import surface_trends


def test_recipe_input_year_error(rootdir):
    test_path = os.path.join(rootdir, 'test_data/')

    recipe_options = {
        'ref_data': test_path,
        'model_name': 'BCC',
        'start_yr': "198012",
        'end_yr': "201042",
        'figure_savepath': 'test_figure',
        'station_code': 'mlo'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_input_model_error(rootdir):
    test_path = os.path.join(rootdir, 'test_data/')

    recipe_options = {
        'ref_data': test_path,
        'model_name': 'BCasdasdjkhgC',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': 'test_figure',
        'station_code': 'mlo'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_input_stationcode_error(rootdir):
    test_path = os.path.join(rootdir, 'test_data/')

    recipe_options = {
        'ref_data': test_path,
        'model_name': 'BCC',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': 'test_figure',
        'station_code': 'asdkjhfasg'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_completes_with_no_errors(rootdir):
    test_path = os.path.join(rootdir, 'test_data/')

    recipe_options = {
        'ref_data': test_path,
        'model_name': 'BCC',
        'start_yr': "1980",
        'end_yr': "2010",
        'figure_savepath': 'test_figure',
        'station_code': 'mlo'}
    try:
        data_dict = surface_trends(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'surface_trends' raised an exception {exc}"
