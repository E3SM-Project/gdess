import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define functions to be imported by *, e.g. from the local __init__ file
#   (also to avoid adding above imports to other namespaces)
__all__ = ['make_my_base_map']


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
    coastline_kw
        dictionary of parameters for the coastlines feature
        default is (color='black', linewidth=0.5)
    borders_kw
        dictionary of parameters for the borders feature
        default is (linestyle=':')
    oceans_kw
        dictionary of parameters for the oceans feature
        default is (facecolor='whitesmoke')
    gridlines_kw
        dictionary of parameters for gridlines
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
