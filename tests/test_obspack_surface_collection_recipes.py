import pytest
from pathlib import Path

import xarray as xr

from gdess import load_stations_dict
from gdess.data_source.observations.gvplus_surface import Collection


@pytest.fixture
def newEmptySurfaceStation():
    mySurfaceInstance = Collection()
    return mySurfaceInstance

@pytest.fixture
def globalview_test_data_path(rootdir: Path):
    return rootdir / 'test_data' / 'globalview'

def test_station_MLO_is_present(newEmptySurfaceStation):
    station_dict = load_stations_dict()
    assert 'mlo' in station_dict


def test_simplest_preprocessed_type(globalview_test_data_path, newEmptySurfaceStation):
    newEmptySurfaceStation.preprocess(datadir=globalview_test_data_path, station_name='mlo')
    assert isinstance(newEmptySurfaceStation.stepA_original_datasets['mlo'], xr.Dataset)


def test_recipe_input_year_error(globalview_test_data_path, newEmptySurfaceStation):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'start_yr': "198012",
        'end_yr': "201042",
        'station_code': 'mlo'}
    with pytest.raises(SystemExit):
        newEmptySurfaceStation.run_recipe_for_timeseries(verbose='DEBUG', options=recipe_options)


def test_recipe_input_stationcode_error(globalview_test_data_path, newEmptySurfaceStation):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'start_yr': "1980",
        'end_yr': "2010",
        'station_code': 'asdkjhfasg'}
    with pytest.raises(SystemExit):
        newEmptySurfaceStation.run_recipe_for_timeseries(verbose='DEBUG', options=recipe_options)


def test_timeseries_recipe_completes_with_no_errors(globalview_test_data_path, newEmptySurfaceStation):
    recipe_options = {
        'ref_data': globalview_test_data_path,
        'start_yr': "1980",
        'end_yr': "2010",
        'station_code': 'mlo'}
    try:
        newEmptySurfaceStation.run_recipe_for_timeseries(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"
