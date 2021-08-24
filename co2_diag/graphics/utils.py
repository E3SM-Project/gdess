import os
from datetime import datetime
import numpy as np

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from co2_diag.formatters import append_before_extension

colormap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colormaps')


def aesthetic_grid_no_spines(axis):
    axis.grid(True, linestyle='--', color='gray', alpha=1)
    for spine in axis.spines.values():
        spine.set_visible(False)


def mysavefig(fig, results_dir='', plot_save_name='test', **kwargs):
    """Append today's date to the file path and save with a tight bbox

    Parameters
    ----------
    fig
    results_dir: str
    plot_save_name: str
    kwargs

    Returns
    -------

    """
    path = os.path.join(results_dir, plot_save_name)
    # If the path doesn't have an extension, we will default to png.
    if not path.lower().endswith(('.png', '.pdf', '.tif', '.tiff', '.jpg', '.jpeg')):
        path += '.png'
    # A datetime stamp is added immediately before the extension.
    path_with_datetime = append_before_extension(path, datetime.today().strftime('%Y-%m-%dT%H%M%S.%f'))

    fig.savefig(path_with_datetime, **kwargs)


def limits_with_zero(t: tuple):
    """Take a 2-tuple of axis limits
    If zero is not between them, then replace the one closest to zero with zero.
    """
    if (not t) | (len(t) > 2):
        raise ValueError("Unexpected input size. Should be a tuple of length 2.")
    elif not all([isinstance(x, (int, float)) for x in t]):
        raise TypeError("Unexpected input types. Should be float or integer.")

    zero_crossings = np.where(np.diff(np.sign(t)))[0]
    if len(zero_crossings) > 0:
        # tuple already crosses or contains zero
        return t
    elif (t[0] > 0) & (t[1] > 0):
        # both are positive
        if t[0] < t[1]:
            return 0, t[1]
        else:
            return t[0], 0
    elif (t[0] < 0) & (t[1] < 0):
        # both are negative
        if t[0] < t[1]:
            return t[0], 0
        else:
            return 0, t[1]
    else:
        raise ValueError("Unexpected condition")


def get_colormap(colormap=None, colormap_search_dir=None):
    if not colormap:
        colormap = "WhiteBlueGreenYellowRed.rgb"
    if not colormap_search_dir:
        colormap_search_dir = colormap_dir

    installed_colormap = os.path.join(colormap_search_dir, colormap)

    try:
        matplotlib_cmap = cm.get_cmap(colormap)
    except ValueError:
        matplotlib_cmap = None
        pass

    if os.path.exists(installed_colormap):
        # use the colormap from ./colormaps
        colormap = installed_colormap
    elif matplotlib_cmap:
        return matplotlib_cmap
    elif not os.path.exists(installed_colormap):
        msg = "File {} isn't installed in {} or in matplotlib"
        raise IOError(msg.format(colormap, colormap_search_dir))

    rgb_arr = np.loadtxt(colormap)
    rgb_arr = rgb_arr / 255.0
    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=rgb_arr)

    return cmap
