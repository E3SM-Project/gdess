import pytest
import os
import tempfile
import pandas as pd
import datacompy

from ccgcrv.ccgcrv import ccgcrv


def test_curvefitting_results_remain_the_same(rootdir):
    expected_results_path = os.path.join(rootdir, 'test_data', 'expected_curvefit_results.txt')
    mlotestdata_path = os.path.join(rootdir, 'test_data', 'mlotestdata.txt')

    with tempfile.TemporaryDirectory() as td:
        output_file_name = os.path.join(td, 'curvefitting_test_results_mlo.txt')

        options = {'npoly': 2,
                   'nharm': 2,
                   'file': output_file_name,
                   'equal': '',
                   'showheader': '',
                   'func': '',
                   'poly': '',
                   'trend': '',
                   'res': ''}
        ccgcrv(options, mlotestdata_path)

        df_expected = pd.read_csv(expected_results_path, sep='\s+')
        df_filter_output = pd.read_csv(output_file_name, sep='\s+')

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
