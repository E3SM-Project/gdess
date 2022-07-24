import pytest

from gdess.formatters import numstr, my_round, \
    tex_escape


def test_number_string_formatting_2decimals():
    assert numstr(3.14159, decimalpoints=2) == '3.14'


def test_number_string_formatting_largenumber():
    assert numstr(3141592.6534, decimalpoints=3) == '3,141,592.653'


def test_rounding_to_nearest_up10():
    assert my_round(23, nearest=10, direction='up') == 30


def test_rounding_to_nearest_down100():
    assert my_round(761523, nearest=100, direction='down') == 761500


def test_rounding_to_nearest_bad_direction():
    with pytest.raises(ValueError):
        my_round(761523, nearest=100, direction='side')


def test_escaping_tex():
    assert tex_escape(r'\$ \_ >') == \
           "\\textbackslash{}\\$ \\textbackslash{}\\_ \\textgreater{}"
