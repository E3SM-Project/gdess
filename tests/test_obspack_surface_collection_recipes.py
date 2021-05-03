import os
import pytest
import xarray as xr

from co2_diag.data_source.obspack.surface_stations.collection import Collection


@pytest.fixture
def newEmptySurfaceStation():
    mySurfaceInstance = Collection()
    return mySurfaceInstance


def test_station_MLO_is_present(newEmptySurfaceStation):
    assert 'mlo' in newEmptySurfaceStation.station_dict


def test_simplest_preprocessed_type(rootdir, newEmptySurfaceStation):
    test_path = os.path.join(rootdir, 'test_data')

    newEmptySurfaceStation.preprocess(datadir=test_path, station_name='mlo')
    assert isinstance(newEmptySurfaceStation.stepA_original_datasets['mlo'], xr.Dataset)


def test_recipe_input_year_error(rootdir, newEmptySurfaceStation):
    test_path = os.path.join(rootdir, 'test_data')

    recipe_options = {
        'ref_data': test_path,
        'start_yr': "198012",
        'end_yr': "201042",
        'station_code': 'mlo'}
    with pytest.raises(SystemExit):
        newEmptySurfaceStation.run_recipe_for_timeseries(verbose='DEBUG', options=recipe_options)


def test_recipe_input_stationcode_error(rootdir, newEmptySurfaceStation):
    test_path = os.path.join(rootdir, 'test_data')

    recipe_options = {
        'ref_data': test_path,
        'start_yr': "1980",
        'end_yr': "2010",
        'station_code': 'asdkjhfasg'}
    with pytest.raises(SystemExit):
        newEmptySurfaceStation.run_recipe_for_timeseries(verbose='DEBUG', options=recipe_options)


def test_timeseries_recipe_completes_with_no_errors(rootdir, newEmptySurfaceStation):
    test_path = os.path.join(rootdir, 'test_data')

    recipe_options = {
        'ref_data': test_path,
        'start_yr': "1980",
        'end_yr': "2010",
        'station_code': 'mlo'}
    try:
        newEmptySurfaceStation.run_recipe_for_timeseries(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"
