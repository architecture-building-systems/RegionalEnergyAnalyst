from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from  plotly  import  tools
import plotly.plotly as py

import os
from plotly.offline import plot
import pandas as pd
from configuration import CONFIG_FILE, DATA_HDD_CDD_PLOTS_FOLDER, DATA_ANALYSIS_PLOTS_FOLDER, DATA_ALLDATA_FILE


def calc_graph(dataframe, output_path, cities, analysis_field):


    traces = []
    for city in cities:
        data = dataframe[dataframe["CITY"]==city]
        x = data[analysis_field]
        y = data["CITY"]
        traces.append(go.Box(x=x, y=y, name=city, orientation='h', boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))

    # add all data
    x = dataframe[analysis_field]
    dataframe["ALL_CITY"] = "96 cities"
    y = dataframe["ALL_CITY"]
    traces.append(go.Box(x=x, y=y, name="96 cities", orientation='h', boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))

    layout = go.Layout(dict(boxmode ='group', plot_bgcolor = "rgb(236,243,247)",
                             font = dict(family='Helvetica, monospace', size=24)), showlegend=False)
    fig = go.Figure(data=traces, layout=layout)
    plot(fig, auto_open=False, filename=output_path + '//' + "boxplot_all_data" + ".html")

    return fig

def calc_histo(dataframe, output_path, cities, analysis_field):
    import plotly.figure_factory as ff
    data = []
    hist_data = []
    group_labels = []
    for city in cities:
        data = dataframe[dataframe["CITY"]==city]
        x = data[analysis_field].values
        y = data["CITY"].values[0]
        hist_data.append(x)
        group_labels.append(y)

    # add all data
    x = dataframe[analysis_field].values
    dataframe["ALL_CITY"] = "96 cities"
    y = dataframe["ALL_CITY"].values[0]
    hist_data.append(x)
    group_labels.append(y)

    fig = ff.create_distplot(hist_data, group_labels, show_hist=False)
    fig['layout'].update(title='Curve and Rug Plot')

    # traces.append(go.Histogram(x=x, y=y, name=city, histnorm='probability'))

    # layout = go.Layout(dict(boxmode ='group', plot_bgcolor = "rgb(236,243,247)",
    #                          font = dict(family='Helvetica, monospace', size=24)), showlegend=False)
    # fig = go.Figure(data=traces, layout=layout)
    plot(fig, auto_open=False, filename=output_path + '//' + "histrogram_all_data" + ".html")

    return fig

def main(data, output_path, cities, analysis_field):
    # get data from every city and transform into data per scenario

    calc_graph(data, output_path, cities, analysis_field)

    calc_histo(data, output_path, cities, analysis_field)



if __name__ == "__main__":
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City']
    output_path = DATA_ANALYSIS_PLOTS_FOLDER
    analysis_field = "LOG_SITE_ENERGY_MWh_yr"
    data  = pd.read_csv(DATA_ALLDATA_FILE)
    main(data, output_path, cities, analysis_field)