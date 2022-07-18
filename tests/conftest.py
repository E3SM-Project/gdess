from pathlib import PurePath, Path
import pytest


@pytest.fixture
def root_testdir():
    return Path(PurePath(__file__)).parent

@pytest.fixture
def root_outputdir():
    return Path(PurePath(__file__)).parents[2] / 'outputs'
