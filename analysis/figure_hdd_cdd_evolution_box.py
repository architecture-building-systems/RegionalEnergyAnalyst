from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from  plotly  import  tools

import os
from plotly.offline import plot
import pandas as pd
from configuration import CONFIG_FILE, DATA_HDD_CDD_PLOTS_FOLDER, DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE, DATA_RAW_BUILDING_TODAY_HDD_FOLDER



def calc_graph(data_frame, output_path, colors, season):

    fig = tools.make_subplots(rows=1, cols=1)
    for year in years_to_map:
        data = data_frame[data_frame.index == str(year)]
        x = data["City"]
        if season =="heating":
            y = data["HDD_18_5_C"]
            trace = go.Box(x=x, y=y, name="HDD for year" + str(year), marker=dict(color=colors[0]))
            fig.append_trace(trace, 1, 1)
        else:
            y = data["CDD_18_5_C"]
            trace = go.Box(x=x, y=y, name="CDD for year" + str(year), marker=dict(color=colors[1]))
            fig.append_trace(trace, 1, 1)

    fig['layout'].update(boxmode='group')#, yaxis=dict(range=[0, 3500]))
    plot(fig, auto_open=False, filename=output_path + '//' + season + ".html")

    return fig

def main(years_to_map, output_path, future_hdd_cdd_file, analysis_fields, colors, cities):
    # get data from every city and transform into data per scenario
    scenarios = future_hdd_cdd_file["Scenario"].unique()

    data_final = pd.DataFrame()
    for i, city in enumerate(cities):
        data_current = pd.read_csv(os.path.join(DATA_RAW_BUILDING_TODAY_HDD_FOLDER, city + ".csv")).set_index(
            "site_year")
        HDD_baseline = data_current.loc[2010, "hdd_18.5C"]
        CDD_baseline = data_current.loc[2010, "cdd_18.5C"]

        data_future_HDD_CDD = future_hdd_cdd_file[future_hdd_cdd_file['City'] == city]
        HDD_baseline = data_future_HDD_CDD.loc[data_future_HDD_CDD["Scenario"]=="A1B_2020"]["HDD_18_5_C"].values[0]
        CDD_baseline = data_future_HDD_CDD.loc[data_future_HDD_CDD["Scenario"]=="A1B_2020"]["CDD_18_5_C"].values[0]

        # heating case
        data_future_HDD_CDD["YEAR"] = [x.split("_", 1)[1] for x in data_future_HDD_CDD["Scenario"].values]
        data_future_HDD_CDD["TODAY_HDD_18_5_C"] = HDD_baseline
        data_future_HDD_CDD["TODAY_CDD_18_5_C"] =  CDD_baseline
        data_future_HDD_CDD.set_index("YEAR", inplace=True)
        data_future_HDD_CDD["SCENARIO_CLASS"] = [x.split("_", 1)[0] for x in data_future_HDD_CDD["Scenario"].values]
        df = data_future_HDD_CDD[["TODAY_HDD_18_5_C", "TODAY_CDD_18_5_C", "HDD_18_5_C", "CDD_18_5_C", "SCENARIO_CLASS", "City"]]

        data_final = pd.concat([data_final, df])

    data_final["HDD_18_5_C"] = (data_final["HDD_18_5_C"]- data_final["TODAY_HDD_18_5_C"])/data_final["TODAY_HDD_18_5_C"]*100
    data_final["CDD_18_5_C"] = (data_final["CDD_18_5_C"]- data_final["TODAY_CDD_18_5_C"])/data_final["TODAY_CDD_18_5_C"]*100

    for season in ["heating", "cooling"]:
        calc_graph(data_final, output_path, colors, season)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'IPCC_scenarios', 'data')
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
    output_path = DATA_HDD_CDD_PLOTS_FOLDER
    future_hdd_cdd_file = pd.read_csv(DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE)
    analysis_fields = ["A1B", "A1B_CDD"] #["B1", "A1B", "A2", "B1_CDD", "A1B_CDD", "A2_CDD"]
    colors = ["rgb(240,75,91)", "rgb(63,192,194)"] #["rgb(240,75,91)","rgb(246,148,143)","rgb(252,217,210)", "rgb(63,192,194)", "rgb(171,221,222)", "rgb(225,242,242)"]
    years_to_map = [2020, 2030, 2040, 2050]

    main(years_to_map, output_path, future_hdd_cdd_file, analysis_fields,colors, cities)