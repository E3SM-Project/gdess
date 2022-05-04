import pytest

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from gdess.graphics.utils import limits_with_zero, get_colormap
from gdess.graphics.mapping import make_my_base_map


def test_limits_with_zero():
    assert (-5, 4) == limits_with_zero((-5, 4))


def test_limits_both_negative():
    assert (0, -7) == limits_with_zero((-5, -7))


def test_limits_both_positive():
    assert (0, 124965) == limits_with_zero((7897, 124965))


def test_limits_floats():
    assert (12.3, 0) == limits_with_zero((12.3, 5.1))


def test_limits_bad_input_type():
    with pytest.raises(TypeError):
        limits_with_zero(('ab', -7))


def test_limits_bad_input_size():
    with pytest.raises(ValueError):
        limits_with_zero((3, -7, 9))


def test_no_name_gets_a_default_colormap():
    assert isinstance(get_colormap(), LinearSegmentedColormap)


def test_get_colormap_type_by_name():
    assert isinstance(get_colormap('WhiteBlueGreenYellowRed.rgb'),
                      LinearSegmentedColormap)


def test_bad_colormap_name_raises_error():
    with pytest.raises(IOError):
        get_colormap('hey')


def test_base_map_returns_correct_types():
    fig, ax = make_my_base_map()
    assert isinstance(fig, plt.Figure) & isinstance(ax, plt.Axes)


def test_base_map_returns_correct_types():
    fig, ax = make_my_base_map()
    assert isinstance(fig, plt.Figure) & isinstance(ax, plt.Axes)
