from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from  plotly  import  tools
import plotly.plotly as py

import os
from plotly.offline import plot
import pandas as pd
from configuration import CONFIG_FILE,NN_MODEL_PREDICTION_FOLDER, DATA_ENERGY_PLOTS_FOLDER,DATA_POST_FUTURE_ENERGY_CONSUMPTION_FILE, DATA_POST_FUTURE_ENERGY_CONSUMPTION_FILE_WITH_EFFICIENCY
import numpy as np


def calc_graph(data_frame, data_frame_with_efficiency, output_path,  colors_hdd, colors_cdd, cities, scenarios):

    titles = cities
    fig = tools.make_subplots(rows=2, cols=5, shared_xaxes=True, shared_yaxes=True, subplot_titles=titles)

    # "blue": "rgb(63,192,194)",
    # "blue_light": "rgb(171,221,222)",
    # "blue_lighter": "rgb(225,242,242)",
    # "yellow": "rgb(255,209,29)",
    # "yellow_light": "rgb(255,225,133)",
    # "yellow_lighter": "rgb(255,243,211)",
    # "brown": "rgb(174,148,72)",
    # "brown_light": "rgb(201,183,135)",
    # "brown_lighter": "rgb(233,225,207)",
    # "purple": "rgb(171,95,127)",
    # "purple_light": "rgb(198,149,167)",
    # "purple_lighter": "rgb(231,214,219)",
    # "green": "rgb(126,199,143)",
    # "green_light": "rgb(178,219,183)",
    # "green_lighter": "rgb(227,241,228)",

    # blue ["rgb(144,202,249)", "rgb(13,71,161)", "rgb(66,165,245)"]
    # blue ligth ["rgb(225,242,242)","rgb(63,192,194)", "rgb(171,221,222)"]
    # colors_all = ["rgb(239,154,154)", "rgb(183,28,28)","rgb(239,83,80)"]
    colors_commercial = ["rgb(144,202,249)", "rgb(13,71,161)", "rgb(66,165,245)"]
    colors_residential= ["rgb(225,242,242)","rgb(63,192,194)", "rgb(171,221,222)"]
    mode = "lines"

    for i, city in enumerate(cities):
        data = data_frame[data_frame["1_city"] == city]
        data_with_efficiency = data_frame_with_efficiency[data_frame_with_efficiency["1_city"] == city]
        if i < 5:
            row = 1
            cols = i + 1
            yaxis = 'y'
        else:
            row = 2
            cols = i - 4
            yaxis = 'y'
        for j, scenario in enumerate(scenarios):

            #data for the total energy consumption
            data2 = data[data["SCENARIO_CLASS"] == scenario]
            # x = data2.index
            # y = data2["EUI_kWh_m2yr"]
            # trace = go.Scatter(x=x, y=y, name="EUI_kWh_m2yr for scenario " + scenario, mode=mode, marker=dict(color=colors_all[j]))
            # fig.append_trace(trace, row, cols)

            x = data2.index
            y = data2["EUI_kWh_m2yr_commercial"]
            trace = go.Scatter(x=x, y=y, name="EUI_kWh_m2yr commercial for scenario " + scenario, mode=mode, marker=dict(color=colors_commercial[j]),
                               yaxis = yaxis)
            fig.append_trace(trace, row, cols)

            x = data2.index
            y = data2["EUI_kWh_m2yr_residential"]
            trace = go.Scatter(x=x, y=y, name="EUI_kWh_m2yr residential for scenario " + scenario, mode=mode, marker=dict(color=colors_residential[j]),
                               yaxis=yaxis)
            fig.append_trace(trace, row, cols)

        print(city, "commercial", round(((data2.loc["2100", "EUI_kWh_m2yr_commercial"] - data2.loc["2010", "EUI_kWh_m2yr_commercial"])/data2.loc["2010", "EUI_kWh_m2yr_commercial"] /9 )*100,0))
        print(city, "residential", round(((data2.loc["2100", "EUI_kWh_m2yr_residential"] - data2.loc["2010", "EUI_kWh_m2yr_residential"])/data2.loc["2010", "EUI_kWh_m2yr_residential"] / 9)*100,0))



    for i in fig['layout']['annotations']:
        i['font'] = dict(family='Helvetica, monospace', size=20)


    fig['layout'].update( plot_bgcolor= "rgb(236,243,247)",
                         font=dict(family='Helvetica, monospace', size=18),
                          yaxis=dict(range=[0, 550])
                          )
    plot(fig, auto_open=False, filename=output_path + '//' + "lines_energy" + ".html")


    return fig

def main(output_path, future_energy_file, future_energy_file_with_efficiency, colors_hdd, colors_cdd, cities, scenarios):
    # get data from every city and transform into data per scenario
    data_final = pd.DataFrame()
    for i, city in enumerate(cities):

        data_future_energy = future_energy_file[future_energy_file['1_city'] == city]
        # heating case
        data_future_energy["YEAR"] = [x.split("_", 1)[1] for x in data_future_energy["scenario"].values]
        data_future_energy.set_index("YEAR", inplace=True)
        data_future_energy["SCENARIO_CLASS"] = [x.split("_", 1)[0] for x in data_future_energy["scenario"].values]
        df = data_future_energy[["energy_MWh", "EUI_kWh_m2yr", "SCENARIO_CLASS", "1_city", "EUI_kWh_m2yr_commercial", "EUI_kWh_m2yr_residential"]]
        data_final = pd.concat([data_final, df])

    data_final_with_efficiency = pd.DataFrame()
    for i, city in enumerate(cities):

        data_future_energy_efficiency = future_energy_file_with_efficiency[future_energy_file_with_efficiency['1_city'] == city]
        # heating case
        data_future_energy_efficiency["YEAR"] = [x.split("_", 1)[1] for x in data_future_energy_efficiency["scenario"].values]
        data_future_energy_efficiency.set_index("YEAR", inplace=True)
        data_future_energy_efficiency["SCENARIO_CLASS"] = [x.split("_", 1)[0] for x in data_future_energy_efficiency["scenario"].values]
        df = data_future_energy_efficiency[["energy_MWh", "EUI_kWh_m2yr", "SCENARIO_CLASS", "1_city"]]
        data_final_with_efficiency = pd.concat([data_final_with_efficiency, df])

    calc_graph(data_final, data_final_with_efficiency, output_path,  colors_hdd, colors_cdd, cities, scenarios)


if __name__ == "__main__":
    model = "log_nn_wd_4L_2var"
    model_path = os.path.join(NN_MODEL_PREDICTION_FOLDER, model)
    output_path = DATA_ENERGY_PLOTS_FOLDER
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values

    future_energy_file = pd.read_csv(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FILE)
    future_energy_file_with_efficiency = pd.read_csv(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FILE_WITH_EFFICIENCY)


    scenarios = ["B1", "A1B", "A2"]
    colors_hdd = ["rgb(239,154,154)", "rgb(183,28,28)","rgb(239,83,80)"]
    colors_cdd = ["rgb(254,220,198)","rgb(245,131,69)","rgb(248,159,109)"]
    years_to_map = [2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]

    main(output_path, future_energy_file, future_energy_file_with_efficiency, colors_hdd, colors_cdd, cities, scenarios)