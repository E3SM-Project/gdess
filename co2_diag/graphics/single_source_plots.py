import pandas as pd
import matplotlib.pyplot as plt


def plot_annual_series(df_anomaly_yearly: pd.DataFrame,
                       df_anomaly_cycle: pd.DataFrame,
                       titlestr: str
                       ) -> (plt.Figure, plt.Axes, tuple):
    """Make timeseries plot with annual anomalies of co2 concentration.

    Returns
    -------
    matplotlib figure
    matplotlib axis
    tuple
        Extra matplotlib artists used for the bounding box (bbox) when saving a figure
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

    # ---- Plot Observations ----
    ax.plot(df_anomaly_yearly, label='annual cycle',
            color='#C0C0C0', linestyle='-', alpha=0.3, marker='.', zorder=-32)
    ax.plot(df_anomaly_cycle['moy'], df_anomaly_cycle['monthly_anomaly_from_year'],
            label='mean annual cycle', marker='o', zorder=10,
            color=(18 / 255, 140 / 255, 126 / 255))  # (255/255, 127/255, 14/255))
    #
    ax.set_ylim((-13, 7))
    #
    ax.set_ylabel('$CO_2$ (ppm)')
    ax.set_xlabel('month')
    ax.set_title(titlestr, fontsize=12)
    #
    #         ax.text(0.02, 0.92, f"{sc.upper()}, {station_dict[sc]['lat']:.1f}, {station_dict[sc]['lon']:.1f}",
    #                     horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.02, 0.06, f"Global, surface level mean",
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    #
    # Define the legend
    handles, labels = ax.get_legend_handles_labels()
    display = (0, len(handles) - 1)
    leg = ax.legend([handle for i, handle in enumerate(handles) if i in display],
                    [label for i, label in enumerate(labels) if i in display],
                    loc='best', fontsize=12)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh._legmarker.set_alpha(1)
    #
    #         ax.grid(linestyle='--', color='lightgray')
    #         for k in ax.spines.keys():
    #             ax.spines[k].set_alpha(0.5)
    bbox_artists = (leg,)

    return fig, ax, bbox_artists