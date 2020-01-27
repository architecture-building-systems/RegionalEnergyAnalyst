from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from plotly import tools
import plotly.plotly as py

import os
from plotly.offline import plot
import pandas as pd
from configuration import DATA_ENTHALPY_GROWTH_PLOTS_FOLDER, DATA_RAW_BUILDING_ENTHALPY_FOLDER, DATA_ENTHALPY_GROWTH_PLOTS_FOLDER
import numpy as np


def main(flag):
    if flag:
        data_1990_2010 = pd.read_csv(
            os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_efficiency_1990_2010.csv"), sep=';').set_index(
            "2_climate_zone")
        data_A1B_2100 = pd.read_csv(
            os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_efficiency_A1B_2100.csv"), sep=';')
        data_A2_100 = pd.read_csv(os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_efficiency_A2_2100.csv"),
                                  sep=';')
        data_B1_2100 = pd.read_csv(os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_efficiency_B1_2100.csv"),
                                   sep=';')
    else:
        data_1990_2010 = pd.read_csv(os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_1990_2010.csv"), sep=';').set_index("2_climate_zone")
        data_A1B_2100 = pd.read_csv(os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_A1B_2100.csv"), sep=';')
        data_A2_100 = pd.read_csv(os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_A2_2100.csv"), sep=';')
        data_B1_2100 = pd.read_csv(os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_B1_2100.csv"), sep=';')
        data_A1B_2050 = pd.read_csv(os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_A1B_2050.csv"), sep=';')
        data_A2_2050 = pd.read_csv(os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_A2_2050.csv"), sep=';')
        data_B1_2050 = pd.read_csv(os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_B1_2050.csv"), sep=';')


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
    graph_variable = ["6_enthalpy_HUM",
                      "7_enthalpy_DEHUM",
                      "4_enthalpy_H",
                      "5_enthalpy_C",
                      "8_enthalpy_total"]

    text_vars = ["15_growth_enthalpy_HUM",
                      "16_growth_enthalpy_DEHUM",
                      "13_growth_enthalpy_H",
                      "14_growth_enthalpy_C",
                      "17_growth_enthalpy_total"]

    titles = ["Humdification",
              "Dehumidificaition",
              "Heating",
              "Cooling",
              "Total thermal energy"]
    pairs_colors = [["rgb(254,220,198)","rgb(248,159,109)", "rgb(245,131,69)"],
                    ["rgb(231,214,219)", "rgb(198,149,167)", "rgb(171,95,127)"],
                    ["rgb(252,217,210)", "rgb(246,148,143)", "rgb(240,75,91)"],
                    ["rgb(144,202,249)", "rgb(66,165,245)", "rgb(13,71,161)"],
                    ["rgb(255,255,255)","rgb(126,127,132)", "rgb(68,76,83)"]]

    # "red": "rgb(240,75,91)",
    # "red_light": "rgb(246,148,143)",


    data_final_2050 = pd.concat([data_A1B_2050, data_A2_2050, data_B1_2050], ignore_index=True).set_index("2_climate_zone")
    data_final_2050["x"] = "0"
    data_final_2100 = pd.concat([data_A1B_2100, data_A2_100, data_B1_2100], ignore_index=True).set_index("2_climate_zone")
    data_final_2100["x"] = "0"
    data_1990_2010["x"] = "0"
    x = []

    for zone, name in zip(zones,zones_names):
        data_final_2050.loc[zone, "x"] = name
        data_final_2100.loc[zone, "x"] = name
        data_1990_2010.loc[zone, "x"] = name


    for grah_var, text_var, colors, title in zip(graph_variable, text_vars, pairs_colors, titles):

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
                    text=str(round(text,0)),
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
                title='Enthalpy Gradient [kJ/kg.day]',
                zeroline=False,
                range=[0,12000]
            ),
            boxmode='group',
        )
        fig = go.Figure(data=data, layout=layout)

        if flag:
            outputfile = "histogram_efficiency_" + grah_var + "_.html"
        else:
            outputfile = "histogram_" + grah_var + "_.html"
        plot(fig, filename=os.path.join(DATA_ENTHALPY_GROWTH_PLOTS_FOLDER, outputfile))



if __name__ == "__main__":

    flag_efficiency = False
    main(flag_efficiency)