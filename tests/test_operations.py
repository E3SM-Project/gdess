import pytest
import numpy as np
import pandas as pd
import xarray as xr

from co2_diag.operations.time import ensure_datetime64_array
from co2_diag.operations.convert import co2_kgfrac_to_ppm

@pytest.fixture
def random_data():
    # Create random data
    np.random.seed(0)
    co2 = 10 * np.random.rand(2, 2, 3)
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]
    time = pd.date_range("2014-09-06", periods=3)
    reference_time = pd.Timestamp("2014-09-05")

    return co2, lon, lat, time, reference_time


@pytest.fixture
def dataarray_withco2(random_data):
    co2, lon, lat, time, reference_time = random_data

    # Initialize a dataarray with multiple dimensions:
    da = xr.DataArray(
        data=co2,
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


@pytest.fixture
def dataset_withco2(random_data):
    co2, lon, lat, time, reference_time = random_data

    # Initialize a dataset:
    ds = xr.Dataset(
        data_vars=dict(
            co2=(["x", "y", "time"], co2),
        ),
        coords=dict(
            lon=(["x", "y"], lon),
            lat=(["x", "y"], lat),
            time=time,
            reference_time=reference_time,
        ),
        attrs=dict(description="Weather related data."),
    )
    return ds


def test_datetime64_conversion_for_dataarray(dataarray_withco2):
    newarray = ensure_datetime64_array(dataarray_withco2)
    assert isinstance(newarray, np.ndarray) & \
           isinstance(newarray[0], np.datetime64)


def test_unit_conversion_dataset(dataset_withco2):
    an_original_value = dataset_withco2['co2'].isel(x=0, y=0, time=0).values

    newdataset = co2_kgfrac_to_ppm(dataset_withco2, co2_var_name='co2')

    a_new_value = newdataset['co2'].isel(x=0, y=0, time=0).values

    assert a_new_value == (an_original_value * (28.9647 / 44.01) * 1e6)


def test_unit_conversion_dataset_bad_varname_raises(dataset_withco2):
    with pytest.raises(Exception):
        co2_kgfrac_to_ppm(dataset_withco2, co2_var_name='carbon dio')
