from collections.abc import Sequence
import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from adjustText import adjust_text

# Define functions to be imported by *, e.g. from the local __init__ file
#   (also to avoid adding above imports to other namespaces)
__all__ = ['make_my_base_map']

# Position and sizes of subplot axes in page coordinates (0 to 1)
# panel = [(0.1691, 0.6810, 0.6465, 0.2258
panel = [(0.1691, 0.01, 0.99, 0.99)]


def make_my_base_map(projection=ccrs.PlateCarree(),
                     coastline_kw: dict = None,
                     borders_kw: dict = None,
                     oceans_kw: dict = None,
                     gridlines_kw: dict = None) -> (plt.Figure, plt.Axes):
    """

    Parameters
    ----------
    projection
        a cartopy projection object
    coastline_kw : dict
        Parameters for the coastlines feature
        default is (color='black', linewidth=0.5)
    borders_kw : dict
        Parameters for the borders feature
        default is (linestyle=':')
    oceans_kw : dict
        Parameters for the oceans feature
        default is (facecolor='whitesmoke')
    gridlines_kw : dict
        Parameters for gridlines
        default is (draw_labels=True, xlocs=np.arange(-180, 180, 90),
                     linestyle='--', color='lightgray', zorder=0)

    Returns
    -------
        a matplotlib figure and (axes.Axes or array of Axes)
    """
    figure, ax = plt.subplots(
        1, 1, figsize=(10, 12),
        # subplot_kw=dict(projection=crs.Orthographic(central_longitude=-100))
        subplot_kw=dict(projection=projection)
    )

    ax.set_global()

    # Coastlines
    if coastline_kw:
        ax.coastlines(**coastline_kw)
        ax.add_feature(cfeature.COASTLINE, **coastline_kw)
    else:
        ax.coastlines(color='black', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE)

    # Borders
    if borders_kw:
        ax.add_feature(cfeature.BORDERS, **borders_kw)
    else:
        ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Oceans
    if oceans_kw:
        ax.add_feature(cfeature.OCEAN, **oceans_kw)
    else:
        ax.add_feature(cfeature.OCEAN, facecolor='whitesmoke')

    # Gridlines
    if gridlines_kw:
        ax.gridlines(**gridlines_kw)
    else:
        ax.gridlines(draw_labels=True, xlocs=np.arange(-180, 180, 90),
                     linestyle='--', color='lightgray', zorder=0)

    # ax.coastlines(color='tab:green', resolution='10m')
    # ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='tab:blue')
    # ax.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='tab:blue')

    return figure, ax


def determine_tick_step(degrees_covered):
    if degrees_covered > 180:
        return 60
    if degrees_covered > 60:
        return 30
    elif degrees_covered > 30:
        return 10
    elif degrees_covered > 20:
        return 5
    else:
        return 1


def add_site_labels(ax: plt.Axes, labels: Sequence, lats: Sequence, lons: Sequence,
                    **kwargs) -> None:
    """Add data point labels
    And adjust them so that they are not overlapping each other or the data points.

    Note: For densely packed aircraft sites, these arguments worked well:
        force_text=(0.1, 1), force_points=(3.2, 3), expand_points=(1.25, 1.25)
        arrowprops=dict(arrowstyle="->", color='b', lw=0.4)

    Parameters
    ----------
    ax : plt.Axes
    labels : Sequence
    lats : Sequence
    lons : Sequence
    kwargs : dict
    """
    texts = []
    for label, lat, lon in zip(labels, lats, lons):
        if lon > 180:
            lon -= 360
        # If the point is too close to the map's edge, matplotlib won't display the annotation, so we need to nudge it.
        if abs(ax.get_ylim()[0] - lat) < 0.5:
            lat += 0.5
        elif abs(ax.get_ylim()[1] - lat) < 0.5:
            lat -= 0.5
        texts.append(ax.annotate(label.upper(), (lon, lat), color='b'))
    adjust_text(texts, ax=ax, only_move={'points': 'xy', 'texts': 'y'}, **kwargs)
