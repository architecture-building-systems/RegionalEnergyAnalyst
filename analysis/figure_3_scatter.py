from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from  plotly  import  tools
import plotly.plotly as py

import os
from plotly.offline import plot
import pandas as pd
from configuration import CONFIG_FILE, DATA_ANALYSIS_PLOTS_FOLDER, DATA_ALLDATA_FILE, DATA_RAW_BUILDING_TODAY_HDD_FOLDER



def calc_graph(data_frame, output_path, cities, y_var, x_var):

    fig = tools.make_subplots(rows=2, cols=5, shared_yaxes=True,  shared_xaxes=True,
                              subplot_titles=cities)

    for i, city in enumerate(cities):
        data = data_frame[data_frame["CITY"] == city]
        if i < 5:
            row = 1
            cols = i +1
            yaxis = 'y1'
        else:
            row = 2
            cols = i - 4
            yaxis = 'y2'
        for color, scenario in zip(["rgb(144,202,249)", "rgb(66,165,245)"],["Commercial", "Residential"]):
            data2 = data[data["BUILDING_CLASS"] == scenario]
            import numpy as np
            x = data2[x_var] #
            y = data2[y_var] #data2["LOG_THERMAL_ENERGY_MWh_yr"] #
            trace = go.Scatter(x=x, y=y, name=scenario, yaxis = yaxis,  mode = 'markers', marker=dict(color=color))
            print(cols)
            fig.append_trace(trace, row, cols)
            #
            # x = data2["LOG_CDD_FLOOR_18_5_C_m2"]
            # trace = go.Scatter(x=x, y=y, name="HDD for scenario " + scenario)
            # fig.append_trace(trace, row, i+1)

    for i in fig['layout']['annotations']:
        i['font'] = dict(family='Helvetica, monospace', size=20)

    fig['layout'].update( plot_bgcolor= "rgb(236,243,247)",
                         font=dict(family='Helvetica, monospace', size=18)
                         )
    plot(fig, auto_open=False, filename=output_path + '//' + "scatter" + ".html")

    return fig

def main(data, output_path, cities, y, x):
    # get data from every city and transform into data per scenario

    calc_graph(data, output_path, cities, y, x)

if __name__ == "__main__":
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City']
    output_path = DATA_ANALYSIS_PLOTS_FOLDER
    y = "LOG_SITE_ENERGY_MWh_yr"
    x = "LOG_THERMAL_ENERGY_MWh_yr"

    data  = pd.read_csv(DATA_ALLDATA_FILE)
    main(data, output_path, cities, y, x)