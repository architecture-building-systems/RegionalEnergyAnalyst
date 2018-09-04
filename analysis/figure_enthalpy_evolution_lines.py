from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from  plotly  import  tools
import plotly.plotly as py

import os
from plotly.offline import plot
import pandas as pd
from configuration import CONFIG_FILE, DATA_HDD_CDD_PLOTS_FOLDER, DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE
import numpy as np


def calc_graph(data_frame, output_path,  colors_hdd, colors_cdd, cities, scenarios):

    titles = np.append(cities[:5], ["", "", "", "", ""])
    titles = np.append(titles, cities[5:])
    titles = np.append(titles, ["", "", "", "", ""])
    fig = tools.make_subplots(rows=4, cols=5, shared_xaxes=True, shared_yaxes=True, subplot_titles=titles)

    colors_srh2=["rgb(231,214,219)", "rgb(198,149,167)", "rgb(171,95,127)"]
    #colors_srh2= ["rgb(126,127,132)", "rgb(68,76,83)", "rgb(35,31,32)"]
    #colors_srh = [ "rgb(254,220,198)","rgb(248,159,109)", "rgb(245,131,69)"]
    colors_srh = ["rgb(255,255,255)","rgb(126,127,132)", "rgb(68,76,83)"]
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

            y = data2["HKKD_H_kJ_kg"]/1000
            trace = go.Scatter(x=x, y=y, name="HDD for scenario" + scenario, mode=mode, marker=dict(color=colors_hdd[j]), yaxis =yaxis)

            fig.append_trace(trace, row, cols)

            y = data2["HKKD_C_kJ_kg"]/1000
            trace = go.Scatter(x=x, y=y, name="CDD for scenario " + scenario, mode=mode, marker=dict(color=colors_cdd[j]), yaxis =yaxis)
            fig.append_trace(trace, row, cols)

            y = data2["HKKD_kJ_kg"]/1000
            trace = go.Scatter(x=x, y=y, name="TDD for scenario " + scenario, mode=mode, marker=dict(color=colors_srh[j]), yaxis =yaxis)
            fig.append_trace(trace, row, cols)

        print(city, "cooling", (data2.loc["2100", "HKKD_H_kJ_kg"] - data2.loc["2010", "HKKD_H_kJ_kg"])/data2.loc["2010", "HKKD_H_kJ_kg"] /9 )
        print(city, "heating", (data2.loc["2100", "HKKD_C_kJ_kg"] - data2.loc["2010", "HKKD_C_kJ_kg"])/data2.loc["2010", "HKKD_C_kJ_kg"] / 9 )
        print(city, "total", (data2.loc["2100", "HKKD_kJ_kg"] - data2.loc["2010", "HKKD_kJ_kg"]) /data2.loc["2010", "HKKD_kJ_kg"]/ 9)
        print(city, "SHR",(data2.loc["2100", "SHR"] - data2.loc["2010", "SHR"]) / data2.loc["2010", "SHR"] / 9)


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
            trace = go.Scatter(x=x, y=y, name="SHR for scenario " + scenario, mode=mode, marker=dict(color=colors_srh2[j]), yaxis =yaxis)
            fig.append_trace(trace, row, cols)

    for i in fig['layout']['annotations']:
        i['font'] = dict(family='Helvetica, monospace', size=20)


    fig['layout'].update( plot_bgcolor= "rgb(236,243,247)",
                         font=dict(family='Helvetica, monospace', size=18),
                         yaxis1= dict(domain = [0.65, 1]),
                         yaxis2 = dict(domain = [0.54, 0.64]),
                         yaxis3 = dict(domain=[0.11, 0.44]),
                         yaxis4=dict(domain=[0.0, 0.10]),)
    plot(fig, auto_open=False, filename=output_path + '//' + "lines_perc_enthalpy" + ".html")


    return fig

def main(years_to_map, output_path, future_hdd_cdd_file,  colors_hdd, colors_cdd, cities, scenarios):
    # get data from every city and transform into data per scenario
    data_final = pd.DataFrame()
    for i, city in enumerate(cities):

        data_future_HDD_CDD = future_hdd_cdd_file[future_hdd_cdd_file['City'] == city]
        HDD_baseline = data_future_HDD_CDD.loc[data_future_HDD_CDD["Scenario"]=="1990_2010"]["HKKD_H_kJ_kg"].values[0]
        CDD_baseline = data_future_HDD_CDD.loc[data_future_HDD_CDD["Scenario"]=="1990_2010"]["HKKD_C_kJ_kg"].values[0]
        SHR_baseline = data_future_HDD_CDD.loc[data_future_HDD_CDD["Scenario"] == "1990_2010"]["SHR"].values[0]

        # heating case
        data_future_HDD_CDD["YEAR"] = [x.split("_", 1)[1] for x in data_future_HDD_CDD["Scenario"].values]
        data_future_HDD_CDD["TODAY_HDD_18_5_C"] = HDD_baseline
        data_future_HDD_CDD["TODAY_CDD_18_5_C"] =  CDD_baseline
        data_future_HDD_CDD["TODAY_SHR"] =  SHR_baseline
        data_future_HDD_CDD.set_index("YEAR", inplace=True)
        data_future_HDD_CDD["SCENARIO_CLASS"] = [x.split("_", 1)[0] for x in data_future_HDD_CDD["Scenario"].values]
        df = data_future_HDD_CDD[["TODAY_HDD_18_5_C", "TODAY_CDD_18_5_C", "TODAY_SHR", "HKKD_H_kJ_kg", "HKKD_C_kJ_kg","HKKD_kJ_kg", "SHR", "SCENARIO_CLASS", "City"]]

        data_final = pd.concat([data_final, df])

    # data_final["HKKD_H_kJ_kg"] = (data_final["HKKD_H_kJ_kg"]- data_final["TODAY_HDD_18_5_C"])/data_final["TODAY_HDD_18_5_C"]*100
    # data_final["HKKD_C_kJ_kg"] = (data_final["HKKD_C_kJ_kg"]- data_final["TODAY_CDD_18_5_C"])/data_final["TODAY_CDD_18_5_C"]*100

    calc_graph(data_final, output_path,  colors_hdd, colors_cdd, cities, scenarios)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'IPCC_scenarios', 'data')
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
    output_path = DATA_HDD_CDD_PLOTS_FOLDER
    future_hdd_cdd_file = pd.read_csv(DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE)
    scenarios = ["B1", "A1B", "A2"]
    colors_hdd = ["rgb(239,154,154)", "rgb(239,83,80)","rgb(183,28,28)"]
    colors_cdd = ["rgb(144,202,249)", "rgb(66,165,245)", "rgb(13,71,161)"]
    years_to_map = [2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]

    main(years_to_map, output_path, future_hdd_cdd_file, colors_hdd, colors_cdd, cities, scenarios)