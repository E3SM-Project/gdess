import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from .utils import get_colormap

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


"""Set up some plotting functions that mimic the E3SM_diag figures

Plotting functions similar to e3sm_diag are defined.
These are based on the functions in: <br>
https://github.com/E3SM-Project/e3sm_diags/blob/master/acme_diags/plot/cartopy/lat_lon_plot.py
"""

def my_lon_lat_plot_like_diag(lon, lat, var, clevels, cmap,
                              colorbarargs=None):
    n = 0

    # Contour levels
    levels = None
    norm = None
    if clevels is None:
        pass
    else:
        if len(clevels) > 0:
            levels = [-1.0e8] + clevels + [1.0e8]
            norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)

    fig = plt.figure(figsize=(12, 8))
    proj = ccrs.PlateCarree(central_longitude=180)
    global_domain = True
    lon_west, lon_east, lat_south, lat_north = (0, 360, -90, 90)

    lon_covered = lon_east - lon_west
    lon_step = determine_tick_step(lon_covered)
    xticks = np.arange(lon_west, lon_east, lon_step)
    # Subtract 0.50 to get 0 W to show up on the right side of the plot.
    # If less than 0.50 is subtracted, then 0 W will overlap 0 E on the left side of the plot.
    # If a number is added, then the value won't show up at all.
    if global_domain:
        xticks = np.append(xticks, lon_east - 0.50)
    lat_covered = lat_north - lat_south
    lat_step = determine_tick_step(lat_covered)
    yticks = np.arange(lat_south, lat_north, lat_step)
    yticks = np.append(yticks, lat_north)

    # Contour plot
    ax = fig.add_axes(panel[n], projection=proj)
    ax.set_extent([lon_west, lon_east, lat_south, lat_north], crs=proj)
    cmap = get_colormap(cmap)
    p1 = ax.contourf(lon, lat, var,
                     transform=ccrs.PlateCarree(),
                     norm=norm,
                     levels=levels,
                     cmap=cmap,
                     extend='both',
                     )

    # Full world would be aspect 360/(2*180) = 1
    ax.set_aspect((lon_east - lon_west) / (2 * (lat_north - lat_south)))
    ax.coastlines(lw=0.3)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(
        zero_direction_label=True, number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=8.0, direction='out', width=1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Color bar
    cbar = fig.colorbar(p1, orientation='horizontal', pad=0.1)
    cbar.ax.xaxis.set_label_position('bottom')
    ticklabel_fmt = "%5.2f"
    if colorbarargs:
        if 'title' in colorbarargs:
            cbar.ax.set_title(f"{colorbarargs['title']}", fontsize=15)
        if 'xlabel' in colorbarargs:
            cbar.ax.set_xlabel(f"{colorbarargs['xlabel']}")
        if 'tickformat' in colorbarargs:
            ticklabel_fmt = colorbarargs['tickformat']
        if 'xticks' in colorbarargs:
            input_ticks = list(colorbarargs['xticks'])
            cbar.set_ticks(input_ticks[:-1])
            cbar.ax.set_xticklabels([(ticklabel_fmt % l) for l in input_ticks], ha='right', rotation=45)


def my_lon_lat_plot_blocks(lon, lat, var, cmap, vmin=None, vmax=None,
                           colorbarargs=None):
    n = 0

    #    # Contour levels
    #     levels = None

    #     norm = None
    #     if clevels:
    #         if len(clevels) > 0:
    #             levels = [-1.0e8] + clevels + [1.0e8]
    #             norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)

    fig = plt.figure(figsize=(12, 8))
    proj = ccrs.PlateCarree(central_longitude=180)
    global_domain = True
    lon_west, lon_east, lat_south, lat_north = (0, 360, -90, 90)

    lon_covered = lon_east - lon_west
    lon_step = determine_tick_step(lon_covered)
    xticks = np.arange(lon_west, lon_east, lon_step)
    # Subtract 0.50 to get 0 W to show up on the right side of the plot.
    # If less than 0.50 is subtracted, then 0 W will overlap 0 E on the left side of the plot.
    # If a number is added, then the value won't show up at all.
    if global_domain:
        xticks = np.append(xticks, lon_east - 0.50)
    lat_covered = lat_north - lat_south
    lat_step = determine_tick_step(lat_covered)
    yticks = np.arange(lat_south, lat_north, lat_step)
    yticks = np.append(yticks, lat_north)

    # Contour plot
    ax = fig.add_axes(panel[n], projection=proj)
    ax.set_extent([lon_west, lon_east, lat_south, lat_north], crs=proj)
    cmap = get_colormap(cmap)
    norm = None
    if (not (vmin is None)) and (not (vmax is None)):
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        p1 = ax.pcolormesh(lon, lat, var,
                           transform=ccrs.PlateCarree(),
                           norm=norm,
                           cmap=cmap
                           )
    else:
        p1 = ax.pcolormesh(lon, lat, var,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap
                           )

    # Full world would be aspect 360/(2*180) = 1
    ax.set_aspect((lon_east - lon_west) / (2 * (lat_north - lat_south)))
    ax.coastlines(lw=0.3)
    print(xticks)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(
        zero_direction_label=True, number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=8.0, direction='out', width=1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Color bar
    if not (norm is None):
        cbar = fig.colorbar(p1, orientation='horizontal', pad=0.1, norm=norm)
    else:
        cbar = fig.colorbar(p1, orientation='horizontal', pad=0.1)
    cbar.ax.xaxis.set_label_position('bottom')
    if colorbarargs:
        if 'title' in colorbarargs:
            cbar.ax.set_title(f"{colorbarargs['title']}", fontsize=15)
        if 'xlabel' in colorbarargs:
            cbar.ax.set_xlabel(f"{colorbarargs['xlabel']}")

    #     if levels is None:
    #         cbar.ax.tick_params(labelsize=9.0, length=0)
    #     else:
    #         maxval = np.amax(np.absolute(levels[1:-1]))
    #         if maxval < 10.0:
    #             fmt = "%5.2f"
    #             pad = 25
    #         elif maxval < 100.0:
    #             fmt = "%5.1f"
    #             pad = 25
    #         else:
    #             fmt = "%6.1f"
    #             pad = 30
    #         cbar.set_ticks(levels[1:-1])
    #         labels = [fmt % l for l in levels[1:-1]]
    #         cbar.ax.set_yticklabels(labels, ha='right')
    #         cbar.ax.tick_params(labelsize=9.0, pad=pad, length=0)
