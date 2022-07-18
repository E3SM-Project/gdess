import pytest
from pathlib import Path
import datetime, tempfile
import numpy as np
import pandas as pd
import datacompy

from ccgcrv.ccgcrv import ccgcrv
from ccgcrv.ccg_dates import datesOk, intDate, \
    getDate, toMonthDay, getDatetime, getTime, dec2date,\
    dateFromDecimalDate, datetimeFromDateAndTime


@pytest.fixture
def curvefilter(rootdir: Path):
    mlotestdata_path = rootdir / 'test_data' / 'mlotestdata.txt'

    with tempfile.TemporaryDirectory() as td:
        output_file_name = Path(td).resolve() / 'curvefitting_test_results_mlo.txt'

        options = {'npoly': 2,
                   'nharm': 2,
                   'file': output_file_name,
                   'equal': '',
                   'showheader': '',
                   'func': '',
                   'poly': '',
                   'trend': '',
                   'res': '',
                   "stats": '',
                   "amp": '',
                   "mm": '',
                   "annual": ''}
        filt = ccgcrv(options, mlotestdata_path)
        df_filter_output = pd.read_csv(output_file_name, sep='\s+')

    return filt, df_filter_output


def test_curvefitting_results_remain_the_same(rootdir, curvefilter):
    expected_results_path = rootdir / 'test_data' / 'expected_curvefit_results.txt'
    df_expected = pd.read_csv(expected_results_path, sep='\s+')

    filt, df_filter_output = curvefilter
    # Test that results are the same to within a millionth of a percent difference
    compare = datacompy.Compare(
        df_expected,
        df_filter_output,
        join_columns='date',
        # abs_tol=0,  # Optional, defaults to 0
        rel_tol=0.000001,  # Optional, defaults to 0
        df1_name='expected',  # Optional, defaults to 'df1'
        df2_name='filter_output'  # Optional, defaults to 'df2'
    )
    # # This will print out a human-readable report summarizing and sampling differences
    # print(compare.report())

    assert compare.matches(ignore_extra_columns=False)


def test_curve_fitting_results_process_with_no_error(curvefilter):
    filt, df_filter_output = curvefilter

    try:
        x0 = filt.xinterp
        y1 = filt.getFunctionValue(x0)
        y2 = filt.getPolyValue(x0)
        y3 = filt.getSmoothValue(x0)
        y4 = filt.getTrendValue(x0)
        filt.getTrendCrossingDates()
        # Seasonal Cycle
        harmonics = filt.getHarmonicValue(x0)
        # residuals from the function
        resid_from_func = filt.resid
        # smoothed residuals
        resid_smooth = filt.smooth
        # trend of residuals
        resid_trend = filt.trend

        filt.stats()
    except Exception as exc:
        assert False, f"'run_recipe_for_timeseries' raised an exception {exc}"


def test_dates_ok_bad_month():
    with pytest.raises(ValueError):
        datesOk(year=2021, month=0, day=10)


def test_dates_ok_bad_day():
    with pytest.raises(ValueError):
        datesOk(year=2021, month=6, day=34)


def test_dates_ok_bad_hour():
    with pytest.raises(ValueError):
        datesOk(year=2021, month=6, day=10, hour=27)


def test_dates_ok_bad_minute():
    with pytest.raises(ValueError):
        datesOk(year=2021, month=6, day=10, minute=61)


def test_dates_ok_bad_second():
    with pytest.raises(ValueError):
        datesOk(year=2021, month=6, day=10, second=61)


def test_integer_from_date():
    assert intDate(year=2021, month=5, day=5) == 2021050500


def test_date_from_integer():
    year, month, day, hour = getDate(2021050500)
    assert (year, month, day, hour) == (2021, 5, 5, 0)


def test_monthday():
    month, day = toMonthDay(2021, 125)  # May 5th, 2021
    assert (month, day) == (5, 5)


def test_monthday_out_of_range():
    with pytest.raises(ValueError):
        toMonthDay(2021, 370)


def test_get_datetime_with_default_separator():
    expected = datetime.datetime(year=2021, month=5, day=5, hour=0, minute=0, second=0)
    returned = getDatetime("2021 05 05")

    assert returned == expected


def test_get_datetime_with_seconds():
    expected = datetime.datetime(year=2021, month=5, day=5, hour=14, minute=10, second=59)
    returned = getDatetime("2021 05 05 14 10 59")

    assert returned == expected


def test_get_datetime_with_semicolon_separator():
    expected = datetime.datetime(year=2021, month=5, day=5, hour=0, minute=0, second=0)
    returned = getDatetime("2021;05;05", sep=';')
    assert returned == expected


def test_get_datetime_with_bad_input():
    with pytest.raises(Exception):
        getDatetime("2021 02 30")


def test_get_datetime_with_poorly_formatted_string():
    with pytest.raises(ValueError):
        getDatetime("2021:02,30")


def test_datetime_from_date_and_time():
    expected = datetime.datetime(year=2021, month=5, day=5, hour=14, minute=10, second=59)
    returned = datetimeFromDateAndTime(d={'year': 2021, 'month': 5, 'day': 5},
                                       t="14:10:59")
    assert returned == expected


def test_get_time():
    expected = datetime.time(hour=13, minute=32, second=59)
    assert getTime("13 32 59") == expected


def test_get_time_with_separator():
    expected = datetime.time(hour=13, minute=32, second=59)
    assert getTime("13;32;59", sep=';') == expected


def test_get_time_poorly_formatted_string():
    with pytest.raises(ValueError):
        getTime("13 3259")


def test_dateFromDecimalDate():
    expected = datetime.date(2021, 5, 5)
    assert dateFromDecimalDate(2021.3411) == expected


def test_decimal_dates():
    # Check equality up to minute precision (don't expect seconds precision)
    assert np.array_equal(dec2date(np.array([2021.3411]))[0, 0:5],
                          np.array([2021, 5, 5, 12, 2])
                          )
