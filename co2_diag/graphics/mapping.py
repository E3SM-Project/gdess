import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define functions to be imported by *, e.g. from the local __init__ file
#   (also to avoid adding above imports to other namespaces)
__all__ = ['make_my_base_map']


def make_my_base_map(projection=ccrs.PlateCarree()):
    figure, ax = plt.subplots(
        1, 1, figsize=(10, 12),
        # subplot_kw=dict(projection=crs.Orthographic(central_longitude=-100))
        subplot_kw=dict(projection=projection)
    )

    ax.set_global()
    ax.coastlines(color='black', linewidth=0.5)

    # ax.coastlines(color='tab:green', resolution='10m')
    # ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='tab:blue')
    # ax.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='tab:blue')

    ax.add_feature(cfeature.OCEAN, facecolor='whitesmoke')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, xlocs=np.arange(-180, 180, 90),
                 linestyle='--', color='lightgray', zorder=0)

    return figure, ax
