from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from  plotly  import  tools
import plotly.plotly as py

import os
from plotly.offline import plot
import pandas as pd
from configuration import CONFIG_FILE, DATA_HDD_CDD_PLOTS_FOLDER, DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE_EFFICEINCY
import numpy as np


def calc_graph(data_frame, output_path, cities, scenarios):

    titles = np.append(cities[:5], ["", "", "", "", ""])
    titles = np.append(titles, cities[5:])
    titles = np.append(titles, ["", "", "", "", ""])
    fig = tools.make_subplots(rows=4, cols=5, shared_xaxes=True, shared_yaxes=True, subplot_titles=titles)

    colors_hdd = ["rgb(239,154,154)", "rgb(239,83,80)","rgb(183,28,28)"]
    colors_cdd = ["rgb(144,202,249)", "rgb(66,165,245)", "rgb(13,71,161)"]

    colors_shr=["rgb(227,241,228)", "rgb(126,199,143)", "rgb(178,219,183)"]
    colors_tot = ["rgb(255,255,255)","rgb(126,127,132)", "rgb(68,76,83)"]

    colors_dehum= ["rgb(231,214,219)", "rgb(198,149,167)", "rgb(171,95,127)"]
    colors_hum = [ "rgb(254,220,198)","rgb(248,159,109)", "rgb(245,131,69)"]

    mode = "lines"
    for i, city in enumerate(cities):
        data = data_frame[data_frame["City"] == city]
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

            y = data2["HKKD_H_sen_kJ_kg"]/1000
            trace = go.Scatter(x=x, y=y, name="HDD for scenario" + scenario, mode=mode, marker=dict(color=colors_hdd[j]), yaxis =yaxis)

            fig.append_trace(trace, row, cols)

            y = data2["HKKD_C_sen_kJ_kg"]/1000
            trace = go.Scatter(x=x, y=y, name="CDD for scenario " + scenario, mode=mode, marker=dict(color=colors_cdd[j]), yaxis =yaxis)
            fig.append_trace(trace, row, cols)

            y = data2["HKKD_HUM_kJ_kg"]/1000
            trace = go.Scatter(x=x, y=y, name="HKKD_HUM_kJ_kg for scenario " + scenario, mode=mode, marker=dict(color=colors_hum[j]), yaxis =yaxis)
            fig.append_trace(trace, row, cols)

            y = data2["HKKD_DEHUM_kJ_kg"]/1000
            trace = go.Scatter(x=x, y=y, name="HKKD_DEHUM_kJ_kg for scenario " + scenario, mode=mode, marker=dict(color=colors_dehum[j]), yaxis =yaxis)
            fig.append_trace(trace, row, cols)

            y = data2["HKKD_kJ_kg"]/1000
            trace = go.Scatter(x=x, y=y, name="TDD for scenario " + scenario, mode=mode, marker=dict(color=colors_tot[j]), yaxis =yaxis)
            fig.append_trace(trace, row, cols)

        print(city, "heating", round(((data2.loc["2100", "HKKD_H_sen_kJ_kg"] - data2.loc["2010", "HKKD_H_sen_kJ_kg"])/data2.loc["2010", "HKKD_H_sen_kJ_kg"] /9 )*100,0))
        print(city, "cooling", round(((data2.loc["2100", "HKKD_C_sen_kJ_kg"] - data2.loc["2010", "HKKD_C_sen_kJ_kg"])/data2.loc["2010", "HKKD_C_sen_kJ_kg"] / 9)*100,0))
        print(city, "humidification", round(((data2.loc["2100", "HKKD_HUM_kJ_kg"] - data2.loc["2010", "HKKD_HUM_kJ_kg"])/data2.loc["2010", "HKKD_HUM_kJ_kg"] /9 )*100,0))
        print(city, "dehumidification", round(((data2.loc["2100", "HKKD_DEHUM_kJ_kg"] - data2.loc["2010", "HKKD_DEHUM_kJ_kg"])/data2.loc["2010", "HKKD_DEHUM_kJ_kg"] / 9)*100,0))
        print(city, "total", round(((data2.loc["2100", "HKKD_kJ_kg"] - data2.loc["2010", "HKKD_kJ_kg"]) /data2.loc["2010", "HKKD_kJ_kg"]/ 9)*100,0))
        print(city, "SHR",round(((data2.loc["2100", "SHR"] - data2.loc["2010", "SHR"]) / data2.loc["2010", "SHR"] / 9)*100,0))


    for i, city in enumerate(cities):
        data = data_frame[data_frame["City"] == city]
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

            y = data2["SHR"]
            trace = go.Scatter(x=x, y=y, name="SHR for scenario " + scenario, mode=mode, marker=dict(color=colors_shr[j]), yaxis =yaxis)
            fig.append_trace(trace, row, cols)

    for i in fig['layout']['annotations']:
        i['font'] = dict(family='Helvetica, monospace', size=20)


    fig['layout'].update( plot_bgcolor= "rgb(236,243,247)",
                         font=dict(family='Helvetica, monospace', size=18),
                         yaxis1= dict(domain = [0.65, 1], range=[0, 12]),
                         yaxis2 = dict(domain = [0.55, 0.63], range=[0, 1]),
                         yaxis3 = dict(domain=[0.10, 0.44], range=[0, 12]),
                         yaxis4=dict(domain=[0.0, 0.08], range=[0, 1]))
    plot(fig, auto_open=False, filename=output_path + '//' + "lines_total_incl_efficiency" + ".html")


    return fig

def main(years_to_map, output_path, future_hdd_cdd_file, cities, scenarios):
    # get data from every city and transform into data per scenario
    data_final = pd.DataFrame()
    for i, city in enumerate(cities):

        data_future_HDD_CDD = future_hdd_cdd_file[future_hdd_cdd_file['City'] == city]

        # heating case
        data_future_HDD_CDD["YEAR"] = [x.split("_", 1)[1] for x in data_future_HDD_CDD["Scenario"].values]
        # data_future_HDD_CDD["TODAY_HDD_18_5_C"] = HDD_baseline
        # data_future_HDD_CDD["TODAY_CDD_18_5_C"] =  CDD_baseline
        # data_future_HDD_CDD["TODAY_SHR"] =  SHR_baseline
        data_future_HDD_CDD.set_index("YEAR", inplace=True)
        data_future_HDD_CDD["SCENARIO_CLASS"] = [x.split("_", 1)[0] for x in data_future_HDD_CDD["Scenario"].values]
        df = data_future_HDD_CDD

        data_final = pd.concat([data_final, df])

    calc_graph(data_final, output_path, cities, scenarios)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'IPCC_scenarios', 'data')
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
    output_path = DATA_HDD_CDD_PLOTS_FOLDER
    future_hdd_cdd_file = pd.read_csv(DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE_EFFICEINCY)
    scenarios = ["B1", "A2", "A1B"]
    years_to_map = [2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]


    main(years_to_map, output_path, future_hdd_cdd_file, cities, scenarios)