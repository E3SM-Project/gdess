from pathlib import PurePath
import pytest


@pytest.fixture
def rootdir():
    return PurePath(__file__).parent
