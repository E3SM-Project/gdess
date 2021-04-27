import pytest

from co2_diag.recipes.utils import options_to_args, valid_year_string, is_some_none


def test_options_to_args():
    options = {'option1': 'val1', 'option2': 2, 'option3': None, 'option4': True}
    expected = ['--option1', 'val1',
                '--option2', '2',
                '--option3', 'None',
                '--option4', 'True']

    assert options_to_args(options) == expected


def test_a_valid_year():
    assert isinstance(valid_year_string('2021'), str)


def test_an_invalid_year():
    with pytest.raises(Exception):
        valid_year_string('20211234')


def test_a_none_year():
    assert valid_year_string(None) is None


def test_a_nonestr_year():
    assert valid_year_string('none') is None


def test_none_is_some_none():
    assert is_some_none(None) is True


def test_nonestr_is_some_none():
    assert is_some_none('None') is True
