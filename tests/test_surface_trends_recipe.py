import pytest
import xarray as xr

from co2_diag.data_source.obspack.surface_stations.collection import Collection
from co2_diag.recipes import surface_trends


@pytest.fixture
def newEmptySurfaceStation():
    mySurfaceInstance = Collection()
    return mySurfaceInstance


def test_station_MLO_is_present(newEmptySurfaceStation):
    assert 'mlo' in newEmptySurfaceStation.station_dict


def test_simplest_preprocessed_type(newEmptySurfaceStation):
    newEmptySurfaceStation.preprocess(datadir='./', station_name='mlo')
    assert isinstance(newEmptySurfaceStation.stepA_original_datasets['mlo'], xr.Dataset)


def test_recipe_input_year_error(newEmptySurfaceStation):
    recipe_options = {
        'ref_data': './',
        'model_name': 'BCC',
        'start_yr': "198012",
        'end_yr': "201042",
        'savepath_figure': 'test_figure',
        'station_code': 'mlo'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_input_model_error(newEmptySurfaceStation):
    recipe_options = {
        'ref_data': './',
        'model_name': 'BCasdasdjkhgC',
        'start_yr': "1980",
        'end_yr': "2010",
        'savepath_figure': 'test_figure',
        'station_code': 'mlo'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_input_stationcode_error(newEmptySurfaceStation):
    recipe_options = {
        'ref_data': './',
        'model_name': 'BCC',
        'start_yr': "1980",
        'end_yr': "2010",
        'savepath_figure': 'test_figure',
        'station_code': 'asdkjhfasg'}
    with pytest.raises(SystemExit):
        surface_trends(verbose='DEBUG', options=recipe_options)


def test_recipe_completes_with_no_errors(newEmptySurfaceStation):
    recipe_options = {
        'ref_data': './',
        'model_name': 'BCC',
        'start_yr': "1980",
        'end_yr': "2010",
        'savepath_figure': 'test_figure',
        'station_code': 'mlo'}
    try:
        data_dict = surface_trends(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'surface_trends' raised an exception {exc}"
