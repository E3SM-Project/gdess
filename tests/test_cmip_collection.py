import pytest

from gdess.data_source.models.cmip.cmip_collection import Collection
from gdess.data_source.models.cmip.cmip_name_utils import matched_model_and_experiment
from gdess.operations.datasetdict import DatasetDict


@pytest.fixture
def newEmptyCMIPCollection():
    myCMIPInstance = Collection()
    return myCMIPInstance


def test_full_model_name_is_valid():
    retval = matched_model_and_experiment('CMIP.BCC.BCC-CSM2-MR.esm-hist.Amon.gn')
    assert retval == 'CMIP.BCC.BCC-CSM2-MR.esm-hist.Amon.gn'

def test_model_source_and_exp_substring_validates_to_full_name():
    retval = matched_model_and_experiment('BCC-CSM2-MR.esm-hist')
    assert retval == 'CMIP.BCC.BCC-CSM2-MR.esm-hist.Amon.gn'

def test_model_source_shortname_with_exp_validates_to_full_name():
    retval = matched_model_and_experiment('BCC.esm-hist')
    assert retval == 'CMIP.BCC.BCC-CSM2-MR.esm-hist.Amon.gn'

def test_model_sourceid_only_is_invalid_and_raises_exception():
    with pytest.raises(ValueError):
        matched_model_and_experiment('BCC-CSM2-MR')

def test_valid_form_but_incorrect_expid_returns_unchanged_input():
    retval = matched_model_and_experiment('BCC-CSM2-MR.fakeexp')
    assert retval == 'BCC-CSM2-MR.fakeexp'


def test_simplest_loading(newEmptyCMIPCollection):
    newEmptyCMIPCollection._load_data(method='pangeo')
    print("testing print...")
    print(newEmptyCMIPCollection)
    print("test print complete.")
    assert isinstance(newEmptyCMIPCollection.stepA_original_datasets, DatasetDict)


def test_recipe_input_year_error(newEmptyCMIPCollection):
    recipe_options = {
        'model_name': 'BCC.esm-hist',
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


def test_timeseries_recipe_completes_with_no_errors(newEmptyCMIPCollection, root_outputdir):
    recipe_options = {
        'model_name': 'BCC.esm-hist',
        'start_yr': "1980",
        'end_yr': "1990",
        'figure_savepath': (root_outputdir / 'test_figure_annual_cmip_series').resolve()}
    try:
        newEmptyCMIPCollection.run_recipe_for_timeseries(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"


def test_zonal_recipe_completes_with_no_errors(newEmptyCMIPCollection, root_outputdir):
    recipe_options = {
        'model_name': 'CMIP.NCAR.CESM2.esm-hist.Amon.gn',
        'member_key': 'r1i1p1f1',
        'start_yr': "1980",
        'end_yr': "1990",
        'figure_savepath': (root_outputdir / 'test_figure_zonal_cmip_series').resolve()}
    try:
        newEmptyCMIPCollection.run_recipe_for_zonal_mean(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"


def test_annual_series_recipe_completes_with_no_errors(newEmptyCMIPCollection, root_outputdir):
    recipe_options = {
        'model_name': 'BCC.esm-hist',
        'start_yr': "1980",
        'end_yr': "1990",
        'cmip_load_method': 'pangeo',
        'figure_savepath': (root_outputdir / 'test_figure_annual_cmip_series').resolve()}
    try:
        newEmptyCMIPCollection.run_recipe_for_annual_series(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"


def test_vertical_profile_recipe_completes_with_no_errors(newEmptyCMIPCollection, root_outputdir):
    recipe_options = {
        'model_name': 'BCC.esm-hist',
        'start_yr': "1980",
        'end_yr': "1990",
        'figure_savepath': (root_outputdir / 'test_figure_vertical_cmip').resolve()}
    try:
        newEmptyCMIPCollection.run_recipe_for_vertical_profile(verbose='DEBUG', options=recipe_options)
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"
