import os
from pathlib import Path
import pytest


@pytest.fixture
def rootdir():
    return Path(__file__).parent.resolve()
        # os.path.dirname(os.path.abspath(__file__))
