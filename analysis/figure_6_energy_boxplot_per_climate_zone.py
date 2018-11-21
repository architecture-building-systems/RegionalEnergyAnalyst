from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from plotly import tools
import plotly.plotly as py

import os
from plotly.offline import plot
import pandas as pd
from configuration import DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, DATA_ENERGY_PLOTS_FOLDER
import numpy as np


def main(flag):
    if flag:
        data_1990_2010 = pd.read_csv(
            os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_enthalpy_days_efficiency_1990_2010.csv"), sep=';').set_index(
            "2_climate_zone")
        data_A1B_2100 = pd.read_csv(
            os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_enthalpy_days_efficiency_A1B_2100.csv"), sep=';')
        data_A2_2100 = pd.read_csv(os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_enthalpy_days_efficiency_A2_2100.csv"),
                                  sep=';')
        data_B1_2100 = pd.read_csv(os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_enthalpy_days_efficiency_B1_2100.csv"),
                                   sep=';')
    else:
        data_1990_2010 = pd.read_csv(os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_consumption1990_2010.csv"), sep=';').set_index("2_climate_zone")
        data_A1B_2100 = pd.read_csv(os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_consumptionA1B_2100.csv"), sep=';')
        data_A2_2100 = pd.read_csv(os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_consumptionA2_2100.csv"), sep=';')
        data_B1_2100 = pd.read_csv(os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_consumptionB1_2100.csv"), sep=';')
        data_A1B_2050 = pd.read_csv(os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_consumptionA1B_2050.csv"), sep=';')
        data_A2_2050 = pd.read_csv(os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_consumptionA2_2050.csv"), sep=';')
        data_B1_2050 = pd.read_csv(os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_consumptionB1_2050.csv"), sep=';')


    zones = [["1A", "2A", "3A"],
             ["2B", "3B"],
             "3C",
             "4A",
             "4B",
             "4C",
             ["5A", "6A"],
             ["5B", "6B"],
              "7",]
    zones_names = ["Hot-humid",
                   "Hot-dry",
                   "Hot-marine",
                   "Mixed-humid",
                   "Mixed-dry",
                   "Mixed-marine",
                   "Cold-humid",
                   "Cold-dry",
                   "Arctic"]

    ratio_commercial_residential_floor = [0.276, 0.343, 0.189, 0.276, 0.343, 0.189, 0.272, 0.272, 0.272]

    graph_variable = ["energy_MWh",
                      "EUI_kWh_m2yr",
                      "EUI_kWh_m2yr_commercial",
                      "EUI_kWh_m2yr_residential"]

    text_vars = ["delta_energy",
                 "delta_eui",
                 "delta_eui_com",
                 "delta_eui_res"]

    titles = ["Energy consumption [MWh/yr]",
              "Energy Use Intensity [kWh/m2.yr]",
              "Energy Use Intensity [kWh/m2.yr]",
              "Energy Use Intensity [kWh/m2.yr]"]
    pairs_colors = [["rgb(239,154,154)", "rgb(183,28,28)","rgb(239,83,80)"],
                    ["rgb(239,154,154)", "rgb(183,28,28)", "rgb(239,83,80)"],
                    ["rgb(144,202,249)", "rgb(66,165,245)", "rgb(13,71,161)"],
                    ["rgb(200,222,222)", "rgb(171,221,222)", "rgb(63,192,194)"]]

    data_final_2050 = pd.concat([data_A1B_2050, data_A2_2050, data_B1_2050], ignore_index=True).set_index(
        "2_climate_zone")
    data_final_2100 = pd.concat([data_A1B_2100, data_A2_2100, data_B1_2100], ignore_index=True).set_index("2_climate_zone")
    data_final_2050["x"] = "0"
    data_final_2100["x"] = "0"
    data_1990_2010["x"] = "0"
    x = []

    for zone, name in zip(zones,zones_names):
        data_final_2050.loc[zone, "x"] = name
        data_final_2100.loc[zone, "x"] = name
        data_1990_2010.loc[zone, "x"] = name
    counter = 0
    for grah_var, text_var, colors, title, ratio in zip(graph_variable, text_vars, pairs_colors, titles, ratio_commercial_residential_floor):
        if counter == 1:
            data_1990_2010[grah_var] = ratio * data_1990_2010["EUI_kWh_m2yr_commercial"] + (1-ratio)* data_1990_2010["EUI_kWh_m2yr_residential"]
            data_final_2050[grah_var] = ratio * data_final_2050["EUI_kWh_m2yr_commercial"] + (1-ratio)* data_final_2050["EUI_kWh_m2yr_residential"]
            data_final_2100[grah_var] = ratio * data_final_2100["EUI_kWh_m2yr_commercial"] + (1-ratio)* data_final_2100["EUI_kWh_m2yr_residential"]
        counter +=1
        trace2 = go.Box(
            y=data_final_2100[grah_var],
            x=data_final_2100["x"],
            name='2100',
            boxpoints = 'all',
            marker=dict(
                color=colors[2]))

        trace1 = go.Box(
            y=data_final_2050[grah_var],
            x=data_final_2050["x"],
            name='2050',
            boxpoints = 'all',
            marker=dict(
                color=colors[1]))

        annotations =[]
        width = 0.04
        for zone in zones:
            data_text = data_final_2100.loc[zone, text_var]
            text = data_text.median()
            annotations.append(
                dict(
                    x=width,
                    y=1.3,
                    xref='paper',
                    yref='paper',
                    text=str(round(text/9,0)), #so we consuder 9 decades
                    showarrow=False))
            width = width + 0.12

        trace0 = go.Box(
            y=data_1990_2010[grah_var],
            x=data_1990_2010["x"],
            name='today',
            boxpoints = 'all',
            marker=dict(
                color=colors[0]
            ))

        data = [trace0, trace1, trace2]
        layout = go.Layout(plot_bgcolor= "rgb(236,243,247)",legend=dict(x=0.90, y=0.95),
                           annotations=annotations,
                             font=dict(family='Helvetica, monospace', size=18),
                           xaxis=dict(categoryorder="array", categoryarray=zones_names),
            yaxis=dict(
                title=title,
                zeroline=False,
                range=[0, 900]
            ),
            boxmode='group',
        )
        fig = go.Figure(data=data, layout=layout)

        if flag:
            outputfile = "boxplot_efficiency_" + grah_var + "_.html"
        else:
            outputfile = "boxplot" + grah_var + "_.html"
        plot(fig, filename=os.path.join(DATA_ENERGY_PLOTS_FOLDER, outputfile))



if __name__ == "__main__":

    flag_efficiency = False
    main(flag_efficiency)