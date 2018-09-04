from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from  plotly  import  tools
import plotly.plotly as py

import os
from plotly.offline import plot
import pandas as pd
from configuration import CONFIG_FILE, DATA_ANALYSIS_PLOTS_FOLDER, DATA_OPTIMAL_TEMPERATURE_FILE



def calc_graph(data_frame, output_path, cities, y_var, x_var):

    fig = tools.make_subplots(rows=1, cols=1, shared_yaxes=True,  shared_xaxes=True)

    for color, scenario in zip(["rgb(144,202,249)", "rgb(10,165,245)","rgb(144,22,249)", "rgb(66,12,245)", "rgb(66,165,40)"],[14.5, 16.5, 18.5, 20.5, 22.5]):
        data2 = data[data["T_C"] == scenario]
        x = data2[x_var]
        y = data2[y_var]
        trace = go.Scatter(x=x, y=y, name=scenario,  mode = 'markers', marker=dict(color=color))
        fig.append_trace(trace, 1, 1)

    for i in fig['layout']['annotations']:
        i['font'] = dict(family='Helvetica, monospace', size=20)

    fig['layout'].update( plot_bgcolor= "rgb(236,243,247)",
                         font=dict(family='Helvetica, monospace', size=18)
                         )
    plot(fig, auto_open=False, filename=output_path + '//' + "figuire_4_errors" + ".html")

    return fig

def main(data, output_path, cities, y, x):
    # get data from every city and transform into data per scenario

    calc_graph(data, output_path, cities, y, x)

if __name__ == "__main__":
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City']
    output_path = DATA_ANALYSIS_PLOTS_FOLDER
    y = "R2"
    x = "MSE"

    data  = pd.read_csv(DATA_OPTIMAL_TEMPERATURE_FILE)
    main(data, output_path, cities, y, x)