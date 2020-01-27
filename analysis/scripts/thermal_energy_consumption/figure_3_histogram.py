from __future__ import division
from __future__ import print_function

import pandas as pd
from plotly import tools
from plotly.offline import plot

from configuration import CONFIG_FILE, DATA_ANALYSIS_PLOTS_FOLDER, DATA_ALLDATA_FILE


def calc_graph(data_frame, output_path, cities, y_var, x_var):
    fig = tools.make_subplots(rows=2, cols=5, shared_yaxes=True, shared_xaxes=True,
                              subplot_titles=cities)

    import plotly.figure_factory as ff

    for i, city in enumerate(cities):
        data = data_frame[data_frame["CITY"] == city]
        if i < 5:
            row = 1
            cols = i + 1
            yaxis = 'y'
        else:
            row = 2
            cols = i - 4
            yaxis = 'y'
        hist_data = []
        for color, scenario in zip(["rgb(144,202,249)", "rgb(66,165,245)"], ["Commercial", "Residential"]):
            data2 = data[data["BUILDING_CLASS"] == scenario]
            hist_data.append(data2[y].values)
        fig_1 = ff.create_distplot(hist_data, group_labels=["Commercial", "Residential"], colors=["rgb(144,202,249)", "rgb(66,165,245)"],
                                   bin_size=.1)
        displot_1 = fig_1['data']
        fig.append_trace(displot_1[0], row, cols)
        fig.append_trace(displot_1[1], row, cols)
        fig.append_trace(displot_1[2], row, cols)
        fig.append_trace(displot_1[3], row, cols)
        fig.append_trace(displot_1[4], row, cols)
        # x = data2["LOG_CDD_FLOOR_18_5_C_m2"]
        # trace = go.Scatter(x=x, y=y, name="HDD for scenario " + scenario)
        # fig.append_trace(trace, row, i+1)

    for i in fig['layout']['annotations']:
        i['font'] = dict(family='Helvetica, monospace', size=20)

    fig['layout'].update(plot_bgcolor="rgb(236,243,247)",
                         font=dict(family='Helvetica, monospace', size=18)
                         )
    plot(fig, auto_open=False, filename=output_path + '//' + "histogram" + ".html")

    return fig


def main(data, output_path, cities, y, x):
    # get data from every city and transform into data per scenario

    calc_graph(data, output_path, cities, y, x)


if __name__ == "__main__":
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City']
    output_path = DATA_ANALYSIS_PLOTS_FOLDER
    y = "LOG_SITE_ENERGY_MWh_yr"
    x = "LOG_HDD_FLOOR_18_5_C_m2"

    data = pd.read_csv(DATA_ALLDATA_FILE)
    main(data, output_path, cities, y, x)
