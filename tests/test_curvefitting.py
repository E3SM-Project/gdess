import pytest
import os
import tempfile

from ccgcrv.ccgcrv import ccgcrv


def test_curvefitting_results_remain_the_same(rootdir):
    expected_results_path = os.path.join(rootdir, 'test_data', 'expected_curvefit_results.txt')
    mlotestdata_path = os.path.join(rootdir, 'test_data', 'mlotestdata.txt')

    with tempfile.TemporaryDirectory() as td:
        output_file_name = os.path.join(td, 'curvefitting_test_results_mlo.txt')

        # with tempfile.NamedTemporaryFile(suffix='.txt', prefix=('curvefitting_test_results_mlo'),
        #                                  delete=False, mode='w+') as temp:
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

        with open(expected_results_path, 'r') as expected:
            with open(output_file_name, 'r') as ccgcrv_output:
                assert expected.read() == ccgcrv_output.read()
