import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt, dates as mdates, ticker, colors as mcolors

from co2_diag.graphics import aesthetic_grid_no_spines, mysavefig


def plot_comparison_against_model(ref_xdata: pd.DataFrame,
                                  ref_ydata: pd.DataFrame,
                                  ref_label_prefix: str,
                                  mdl_xdata: pd.DataFrame,
                                  mdl_ydata: pd.DataFrame,
                                  mdl_label_prefix: str,
                                  savepath=None) -> None:

    ref_custom_cycler = cycler(color=['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02'])
    mdl_custom_cycler = cycler(color=['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae'])

    # --- Plot the seasonal cycle
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # -- REF --
    if len(ref_ydata.columns) > 1:
        ax.set_prop_cycle(ref_custom_cycler)
        line_props = {'marker': 'o', 'linestyle': '-'}
    else:
        line_props = {'marker': 'o', 'linestyle': '-', 'color': 'black'}
    #
    for y_column_name, y_arr in ref_ydata.iteritems():
        plt.plot(ref_xdata, y_arr, label=f"{ref_label_prefix} [{y_column_name}]", **line_props)
    # -- MDL --
    if len(ref_ydata.columns) > 1:
        ax.set_prop_cycle(mdl_custom_cycler)
        line_props = {'marker': 'o', 'linestyle': '-'}
    else:
        line_props = {'marker': 'o', 'linestyle': '-', 'color': 'red'}
    #
    for y_column_name, y_arr in mdl_ydata.iteritems():
        plt.plot(mdl_xdata, y_arr, label=f"{mdl_label_prefix} [{y_column_name}]", **line_props)
    #
    # --- Set figure properties
    plt.title('annual climatology')
    ax.set_ylabel("$CO_2$ (ppm)")
    #
    # Specify the xaxis tick labels format -- %b gives us Jan, Feb...
    month_fmt = mdates.DateFormatter('%b')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda z, pos=None: month_fmt(z)[0]))
    #
    aesthetic_grid_no_spines(ax)
    #
    lgd = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    #
    plt.tight_layout()
    if savepath:
        mysavefig(fig=fig, plot_save_name=savepath, bbox_inches='tight', bbox_extra_artists=(lgd, ))


def plot_heatmap_of_all_stations(xdata: pd.DataFrame,
                                 ydata: pd.DataFrame,
                                 rightside_labels: list = None,
                                 figure_title: str = '',
                                 savepath=None) -> None:
    mindate = mdates.date2num(xdata.tolist()[0])
    maxdate = mdates.date2num(xdata.tolist()[-1])

    num_stations = ydata.shape[1]
    station_labels = list(ydata.columns.values)

    # --- Plot the seasonal cycle for all stations,
    #   and flip the ydata because pyplot.imshow will plot the last row on the bottom
    totalfigheight = (num_stations*0.8) + (num_stations*0.8)*0.15  # Add 0.15 for the colorbar
    fig, ax = plt.subplots(1, 1, figsize=(6, totalfigheight))
    im = ax.imshow(ydata.transpose().iloc[::-1],
                   norm=mcolors.TwoSlopeNorm(vcenter=0.), cmap='RdBu_r', interpolation='nearest',
                   aspect='auto', extent=(mindate, maxdate, -0.5, num_stations - 0.5))
    #
    ax.set_yticks(range(num_stations))
    ax.set_yticklabels(station_labels)
    #
    # Add secondary y-axis on the right side to show station latitudes
    if rightside_labels:
        ax2 = ax.twinx()
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(num_stations))
        ax2.set_yticklabels(rightside_labels)
    #
    cbar = fig.colorbar(im, orientation="horizontal", pad=max([0.5, (num_stations - 0.5)*0.008]))
    cbar.ax.set_xlabel('$CO_2$ (ppm)')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='vertical')
    #
    # Specify the xaxis tick labels format -- %b gives us Jan, Feb...
    month_fmt = mdates.DateFormatter('%b')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda z, pos=None: month_fmt(z)[0]))
    #
    if figure_title:
        ax.set_title(figure_title)
    #
    if savepath:
        mysavefig(fig=fig, plot_save_name=savepath, bbox_inches='tight')


def plot_lines_for_all_station_cycles(xdata: pd.DataFrame,
                                      ydata: pd.DataFrame,
                                      figure_title: str = '',
                                      savepath=None) -> None:
    # --- Plot the seasonal cycle for all stations
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(xdata, ydata, '-o')
    #
    aesthetic_grid_no_spines(ax)
    #
    lgd = plt.legend(ydata.columns.values, loc='upper left')
    #
    ax.set_ylabel('$CO_2$ (ppm)')
    if figure_title:
        ax.set_title(figure_title)
    #
    # Specify the xaxis tick labels format -- %b gives us Jan, Feb...
    month_fmt = mdates.DateFormatter('%b')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda z, pos=None: month_fmt(z)[0]))
    #
    plt.tight_layout()
    if savepath:
        mysavefig(fig=fig, plot_save_name=savepath, bbox_inches='tight', bbox_extra_artists=(lgd, ))