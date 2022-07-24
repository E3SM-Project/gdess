from pathlib import PurePath, Path
import pytest


@pytest.fixture
def root_testdir():
    return Path(PurePath(__file__)).resolve().parent

@pytest.fixture
def globalview_test_data_path(root_testdir):
    return root_testdir / 'test_data' / 'globalview'

@pytest.fixture
def root_outputdir():
    return Path(PurePath(__file__)).resolve().parents[2] / 'outputs'
