import pytest
import os

from co2_diag.data_source.e3sm import Collection as co2e3sm
from co2_diag.data_source.datasetdict import DatasetDict


@pytest.fixture
def newEmptyE3SMCollection():
    myE3SMInstance = co2e3sm()
    return myE3SMInstance


def test_obj_attributes_return_type(newEmptyE3SMCollection):
    attribute_strings = newEmptyE3SMCollection._obj_attributes_list_str()
    assert isinstance(attribute_strings, list) and (len(attribute_strings) > 0)


def test_simplest_preprocessed_type(rootdir, newEmptyE3SMCollection):
    test_path = os.path.join(rootdir, 'test_data', 'test_co2_hist_files_ne4pg2_2yrbudget_record.CO2.nc')

    newEmptyE3SMCollection.preprocess(filepath=test_path)
    assert isinstance(newEmptyE3SMCollection.stepA_original_datasets, DatasetDict)


def test_recipe_input_year_error(rootdir):
    test_path = os.path.join(rootdir, 'test_data', 'test_co2_hist_files_ne4pg2_2yrbudget_record.CO2.nc')

    recipe_options = {
        'test_data': test_path,
        'start_yr': "198012",
        'end_yr': "201042"}
    with pytest.raises(SystemExit):
        co2e3sm.run_recipe_for_timeseries(verbose='DEBUG', options=recipe_options)


def test_recipe_error_when_passed_invalid_date_option(rootdir):
    test_path = os.path.join(rootdir, 'test_data', 'test_co2_hist_files_ne4pg2_2yrbudget_record.CO2.nc')

    recipe_options = {
        'test_data': test_path,
        'start_yr': "1970"}  # Note: The test output from e3sm only goes from 1950 to 1952.
    with pytest.raises(ValueError):
        co2e3sm.run_recipe_for_timeseries(load_from_file=False, verbose='INFO', options=recipe_options)


def test_recipe_completes_with_no_errors(rootdir):
    test_path = os.path.join(rootdir, 'test_data', 'test_co2_hist_files_ne4pg2_2yrbudget_record.CO2.nc')

    recipe_options = {
        'test_data': test_path,
        'start_yr': "1950"}
    try:
        co2e3sm.run_recipe_for_timeseries(load_from_file=False,
                                          verbose='DEBUG',
                                          options=recipe_options)
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"
