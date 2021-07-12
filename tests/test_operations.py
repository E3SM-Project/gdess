from co2_diag.operations.time import ensure_datetime64_array, ensure_cftime_array, monthlist, dt2t
from co2_diag.operations.convert import co2_kgfrac_to_ppm
from co2_diag.operations.utils import print_var_summary, assert_expected_dimensions
from co2_diag.operations.Confrontation import extract_site_data_from_dataset
import numpy as np
import pandas as pd
import xarray as xr
import cftime, pytest, logging


@pytest.fixture
def random_data():
    # Create random data
    np.random.seed(0)
    co2 = 10 * np.random.rand(5, 5, 3, 4)
    co2[1, 1, 0, [0, 2]] = np.nan
    zg = [50, 75, 100] + (10 * np.random.rand(5, 5, 4, 3))
    zg = np.swapaxes(zg, 2, 3)
    lon = [-180, -90, 0, 90, 180]
    lat = [-90, -45, 0, 45, 90]
    plev = [50, 75, 100]
    time = pd.date_range("2014-09-06", periods=4)
    reference_time = pd.Timestamp("2014-09-05")

    return co2, zg, lon, lat, time, plev, reference_time


@pytest.fixture
def dataarray_withco2(random_data):
    co2, _, lon, lat, time, plev, reference_time = random_data

    # Initialize a dataarray with multiple dimensions:
    da = xr.DataArray(
        data=co2,
        dims=["lon", "lat", "plev", "time"],
        coords=dict(
            lon=lon,
            lat=lat,
            plev=plev,
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
def dataset_withco2andzg(random_data):
    co2, zg, lon, lat, time, plev, reference_time = random_data

    # Initialize a dataset:
    ds = xr.Dataset(
        data_vars=dict(
            co2=(["lon", "lat", "plev", "time"], co2),
            zg=(["lon", "lat", "plev", "time"], zg),
        ),
        coords=dict(
            lon=lon,
            lat=lat,
            plev=plev,
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


def test_cftime_conversion(dataarray_withco2):
    new_time_array = ensure_cftime_array(dataarray_withco2)

    assert all([isinstance(x, cftime.DatetimeGregorian)
                for x in new_time_array])


def test_month_list_generation():
    expected = [np.datetime64(x)
                for x in
                ['2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01']
                ]
    assert monthlist(['2020-11-01', '2021-05-05']) == expected


def test_unit_conversion_dataset(dataset_withco2andzg):
    an_original_value = dataset_withco2andzg['co2'].isel(lon=0, lat=0, plev=0, time=0).values

    newdataset = co2_kgfrac_to_ppm(dataset_withco2andzg, co2_var_name='co2')

    a_new_value = newdataset['co2'].isel(lon=0, lat=0, plev=0, time=0).values

    assert a_new_value == (an_original_value * (28.9647 / 44.01) * 1e6)


def test_unit_conversion_dataset_bad_varname_raises(dataset_withco2andzg):
    with pytest.raises(Exception):
        co2_kgfrac_to_ppm(dataset_withco2andzg, co2_var_name='carbon dio')


def test_conversion_of_datetime_to_a_decimalyear_time():
    afloat = dt2t(year=2013, month=3, day=15, h=12)
    assert (afloat - 2013.20136) < 0.00001


def test_expected_dimension_names_raises_assertionerror(dataset_withco2andzg, caplog):
    with pytest.raises(AssertionError):
        assert_expected_dimensions(dataset_withco2andzg, expected_dims=['beebop'])


def test_expected_dimensions_shape_raises_assertionerror(dataset_withco2andzg, caplog):
    with pytest.raises(AssertionError):
        assert_expected_dimensions(dataset_withco2andzg, expected_dims=['lat', 'plev', 'lon', 'time'],
                                   expected_shape=[5, 90, 3, 3])


def test_expected_dimensions_validates_with_list(dataset_withco2andzg, caplog):
    assert assert_expected_dimensions(dataset_withco2andzg,
                                      expected_dims=['lat', 'plev', 'lon', 'time'],
                                      expected_shape=[5, 3, 5, 4])


def test_expected_dimensions_validates_with_dict(dataset_withco2andzg, caplog):
    assert assert_expected_dimensions(dataset_withco2andzg,
                                      expected_dims=['lat', 'plev', 'lon', 'time'],
                                      expected_shape={'lat': 5, 'time': 4, 'lon': 5, 'plev': 3})


def test_print_netcdf_var_summary(dataset_withco2andzg, caplog):
    print_var_summary(dataset_withco2andzg, varname='co2', return_dataset=False)
    for record in caplog.records:
        assert record.levelname == "INFO"


def test_extract_site_data(dataset_withco2andzg):
    da = extract_site_data_from_dataset(dataset_withco2andzg, lon=-99.32, lat=42.21, drop=True)
    assert assert_expected_dimensions(da,
                                      expected_dims=['plev', 'time'],
                                      expected_shape={'plev': 3, 'time': 4})
