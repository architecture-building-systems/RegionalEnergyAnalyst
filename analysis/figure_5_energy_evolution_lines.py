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

    titles = np.append(cities[:5], ["", "", "", "", ""])
    titles = np.append(titles, cities[5:])
    titles = np.append(titles, ["", "", "", "", ""])
    fig = tools.make_subplots(rows=4, cols=5, shared_xaxes=True, shared_yaxes=True, subplot_titles=titles)

    # colors_srh2=["rgb(231,214,219)", "rgb(171,95,127)", "rgb(198,149,167)"]
    # #colors_srh2= ["rgb(126,127,132)", "rgb(68,76,83)", "rgb(35,31,32)"]
    # # colors_srh = [ "rgb(254,220,198)","rgb(248,159,109)", "rgb(245,131,69)"]
    # colors_srh = ["rgb(255,255,255)", "rgb(68,76,83)", "rgb(126,127,132)"]

    colors_srh2 = ["rgb(239,154,154)", "rgb(183,28,28)","rgb(239,83,80)"]
    colors_srh = ["rgb(254,220,198)","rgb(245,131,69)","rgb(248,159,109)"]
    mode = "lines"
    for i, city in enumerate(cities):
        data = data_frame[data_frame["1_city"] == city]
        data_with_efficiency = data_frame_with_efficiency[data_frame_with_efficiency["1_city"] == city]
        if i < 5:
            row = 1
            cols = i + 1
            yaxis = 'y1'
        else:
            row = 3
            cols = i - 4
            yaxis = 'y3'
        for j, scenario in enumerate(scenarios):
            data2 = data[data["SCENARIO_CLASS"] == scenario]
            x = data2.index
            y = data2["energy_MWh"] # in MWh
            trace = go.Scatter(x=x, y=y, name="energy_MWh for scenario" + scenario, mode=mode, marker=dict(color=colors_hdd[j]), yaxis =yaxis)
            fig.append_trace(trace, row, cols)

            # data2 = data_with_efficiency[data_with_efficiency["SCENARIO_CLASS"] == scenario]
            # x = data2.index
            # y = data2["energy_MWh"] # in MWh
            # trace = go.Scatter(x=x, y=y, name="energy_efficency_MWh for scenario" + scenario, mode=mode, marker=dict(color=colors_cdd[j]), yaxis =yaxis)
            # fig.append_trace(trace, row, cols)


        print(city, "total change in %", (data2.loc["2100", "energy_MWh"] - data2.loc["2010", "energy_MWh"]) /data2.loc["2010", "energy_MWh"]/ 9*100)


    for i, city in enumerate(cities):
        data = data_frame[data_frame["1_city"] == city]
        data_with_efficiency = data_frame_with_efficiency[data_frame_with_efficiency["1_city"] == city]
        if i < 5:
            row = 2
            cols = i + 1
            yaxis = 'y2'
        else:
            row = 4
            cols = i - 4
            yaxis = 'y4'
        for j, scenario in enumerate(scenarios):
            data2 = data[data["SCENARIO_CLASS"] == scenario]
            x = data2.index
            y = data2["EUI_kWh_m2yr"]
            trace = go.Scatter(x=x, y=y, name="EUI_kWh_m2yr for scenario " + scenario, mode=mode, marker=dict(color=colors_srh2[j]), yaxis =yaxis)
            fig.append_trace(trace, row, cols)

            data2 = data_with_efficiency[data_with_efficiency["SCENARIO_CLASS"] == scenario]
            x = data2.index
            y = data2["EUI_kWh_m2yr"]
            trace = go.Scatter(x=x, y=y, name="EUI_kWh_m2yr with efficiency for scenario " + scenario, mode=mode, marker=dict(color=colors_srh[j]), yaxis =yaxis)
            fig.append_trace(trace, row, cols)

    for i in fig['layout']['annotations']:
        i['font'] = dict(family='Helvetica, monospace', size=20)


    fig['layout'].update( plot_bgcolor= "rgb(236,243,247)",
                         font=dict(family='Helvetica, monospace', size=18),
                             yaxis1= dict(domain = [0.78, 1], range=[0, 9000]),
                         yaxis2 = dict(domain = [0.54, 0.74], range=[0, 600]),
                         yaxis3 = dict(domain=[0.24, 0.44], range=[0, 9000]),
                         yaxis4=dict(domain=[0.0, 0.20], range=[0, 600]),)
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
        df = data_future_energy[["energy_MWh", "EUI_kWh_m2yr", "SCENARIO_CLASS", "1_city"]]
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