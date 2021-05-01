import pytest
import numpy as np
import pandas as pd
import xarray as xr

from co2_diag.operations.time import ensure_datetime64_array

@pytest.fixture
def newDataArray():
    # Create random data
    np.random.seed(0)
    temperature = 15 + 8 * np.random.randn(2, 2, 3)
    precipitation = 10 * np.random.rand(2, 2, 3)
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]
    time = pd.date_range("2014-09-06", periods=3)
    reference_time = pd.Timestamp("2014-09-05")

    # Initialize a dataarray with multiple dimensions:
    da = xr.DataArray(
        data=temperature,
        dims=["x", "y", "time"],
        coords=dict(
            lon=(["x", "y"], lon),
            lat=(["x", "y"], lat),
            time=time,
            reference_time=reference_time,
        ),
        attrs=dict(
            description="Ambient temperature.",
            units="degC",
        ),
    )
    return da


def test_datetime64_conversion_for_dataarray(newDataArray):
    newarray = ensure_datetime64_array(newDataArray)
    assert isinstance(newarray, np.ndarray) & \
           isinstance(newarray[0], np.datetime64)
