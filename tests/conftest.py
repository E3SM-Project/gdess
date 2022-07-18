from pathlib import PurePath, Path
import pytest


@pytest.fixture
def rootdir():
    return Path(PurePath(__file__).parent).resolve()
