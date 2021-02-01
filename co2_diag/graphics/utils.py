import os
from datetime import datetime
import numpy as np

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


def aesthetic_grid_no_spines(axis):
    axis.grid(True, linestyle='--', color='gray', alpha=1)
    for spine in axis.spines.values():
        spine.set_visible(False)


def mysavefig(fig, results_dir, plot_save_name, bbox_artists):
    today_str = datetime.today().strftime('%Y-%m-%d')
    fig.savefig(results_dir + plot_save_name + '_' + today_str,
                bbox_inches='tight', bbox_extra_artists=bbox_artists)


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