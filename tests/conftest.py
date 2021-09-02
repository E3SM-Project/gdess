import os
import pytest
from pathlib import Path
import subprocess

test_dir = os.path.dirname(os.path.abspath(__file__))
pathing_script = os.path.join(test_dir, 'set_path_vars_for_testing.sh')

# Set env variable to be visible in this process + all children
os.environ['GDESS_REPO'] = str(Path(test_dir).parent.absolute())

subprocess.call(['sh', f"./{pathing_script}"])


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))
