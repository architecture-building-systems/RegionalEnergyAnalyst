from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from  plotly  import  tools
import plotly.plotly as py

import os
from plotly.offline import plot
import pandas as pd
from configuration import CONFIG_FILE, DATA_RAW_BUILDING_ENTHALPY_FOLDER, DATA_ENTHALPY_GROWTH_PLOTS_FOLDER
import numpy as np


def calc_graph(data_frame, output_path, cities, x_field, y_field):

    y = data_frame[y_field]
    x = data_frame[x_field]
    trace = go.Scatter(x=x, y=y, mode='markers', text=data_frame["1_city"])
    fig = go.Figure(data=[trace])
    fig['layout'].update(plot_bgcolor= "rgb(236,243,247)",
                         font=dict(family='Helvetica, monospace', size=18),
                         yaxis=dict(tick0=-10),
                         xaxis = dict(tick0=-10))
    plot(fig, auto_open=False, filename=output_path + '//' + "x_field" + ".html")


    return fig

def main(output_path, cities, scenario):
    # get data from every city and transform into data per scenario
    data_final = pd.read_csv(os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_"+scenario+".csv"), sep=";")
    enthalpy = ["4_enthalpy_H"]
    perc_enthalpy = ["13_per_enthalpy_H"]
    for x_field, y_field in zip(enthalpy, perc_enthalpy):
        calc_graph(data_final, output_path, cities, x_field, y_field)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'IPCC_scenarios', 'data')
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
    output_path = DATA_ENTHALPY_GROWTH_PLOTS_FOLDER
    scenario = "A1B_2100"

    main(output_path, cities, scenario)