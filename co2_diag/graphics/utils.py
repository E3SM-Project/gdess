import os
from datetime import datetime
import numpy as np

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


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
    def append_date_before_extension(filename):
        today_str = datetime.today().strftime('%Y-%m-%d')
        return "{0}_{2}{1}".format(*os.path.splitext(filename) + (today_str,))

    fig.savefig(append_date_before_extension(results_dir + plot_save_name),
                bbox_inches='tight', **kwargs)


def limits_with_zero(t: tuple):
    """Take a 2-tuple of axis limits
    If zero is not between them, then replace the one closest to zero with zero.

    Note: we assume that the value closer to zero will be first (last) when the values are both positive (negative)
    """
    if (not t) | (len(t) > 2) | (not all([isinstance(x, (int, float)) for x in t])):
        raise ValueError("Unexpected input format")

    zero_crossings = np.where(np.diff(np.sign(t)))[0]
    if len(zero_crossings) > 0:
        return t  # tuple already crosses or contains zero
    elif (t[0] > 0) & (t[1] > 0):
        return 0, t[1]
    elif (t[0] < 0) & (t[1] < 0):
        return t[0], 0
    else:
        raise ValueError("Unexpected condition")


def get_colormap(colormap, colormap_search_dir):
    if not colormap:
        colormap = "WhiteBlueGreenYellowRed.rgb"
    if not colormap_search_dir:
        colormap_search_dir = '/global/homes/d/dekauf/colormaps/'

    installed_colormap = os.path.join(colormap_search_dir, colormap)

    try:
        matplotlib_cmap = cm.get_cmap(colormap)
    except ValueError:
        matplotlib_cmap = None
        pass

    if os.path.exists(colormap):
        # colormap is an .rgb in the current directory
        pass
    elif not os.path.exists(colormap) and os.path.exists(installed_colormap):
        # use the colormap from /plot/colormaps
        colormap = installed_colormap
    elif matplotlib_cmap:
        return matplotlib_cmap
    elif not os.path.exists(colormap) and not os.path.exists(installed_colormap):
        pth = os.path.join(colormap_search_dir, 'colormaps')
        msg = "File {} isn't in the current working directory or installed in {}"
        raise IOError(msg.format(colormap, pth))

    rgb_arr = np.loadtxt(colormap)
    rgb_arr = rgb_arr / 255.0
    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=rgb_arr)

    return cmap