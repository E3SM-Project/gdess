import os
from pathlib import PurePath
import pytest


@pytest.fixture
def rootdir():
    return PurePath(__file__).parent
        # os.path.dirname(os.path.abspath(__file__))
