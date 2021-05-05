import pytest

from co2_diag.data_source.cmip.collection import Collection
from co2_diag.data_source.datasetdict import DatasetDict


@pytest.fixture
def newEmptyCMIPCollection():
    myCMIPInstance = Collection()
    return myCMIPInstance


def test_simplest_preprocessed_type(newEmptyCMIPCollection):
    newEmptyCMIPCollection.preprocess()
    assert isinstance(newEmptyCMIPCollection.stepA_original_datasets, DatasetDict)


def test_recipe_input_year_error(newEmptyCMIPCollection):
    recipe_options = {
        'model_name': 'BCC',
        'start_yr': "198012",
        'end_yr': "201042"}
    with pytest.raises(SystemExit):
        newEmptyCMIPCollection.run_recipe_for_timeseries(verbose='DEBUG', options=recipe_options)


def test_recipe_input_model_error(newEmptyCMIPCollection):
    recipe_options = {
        'model_name': 'BCasdasdjkhgC',
        'start_yr': "1980",
        'end_yr': "2010"}
    with pytest.raises(SystemExit):
        newEmptyCMIPCollection.run_recipe_for_timeseries(verbose='DEBUG', options=recipe_options)


def test_timeseries_recipe_completes_with_no_errors(newEmptyCMIPCollection):
    recipe_options = {
        'model_name': 'BCC',
        'start_yr': "1980",
        'end_yr': "1990"}
    try:
        newEmptyCMIPCollection.run_recipe_for_timeseries(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"


def test_zonal_recipe_completes_with_no_errors(newEmptyCMIPCollection):
    recipe_options = {
        'model_name': 'CMIP.NCAR.CESM2.esm-hist.Amon.gn',
        'member_key': 'r1i1p1f1',
        'start_yr': "1980",
        'end_yr': "1990"}
    try:
        newEmptyCMIPCollection.run_recipe_for_zonal_mean(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"


def test_annual_series_recipe_completes_with_no_errors(newEmptyCMIPCollection):
    recipe_options = {
        'model_name': 'BCC',
        'start_yr': "1980",
        'end_yr': "1990"}
    try:
        newEmptyCMIPCollection.run_recipe_for_annual_series(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"


def test_vertical_profile_recipe_completes_with_no_errors(newEmptyCMIPCollection):
    recipe_options = {
        'model_name': 'BCC',
        'start_yr': "1980",
        'end_yr': "1990"}
    try:
        newEmptyCMIPCollection.run_recipe_for_vertical_profile(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"
