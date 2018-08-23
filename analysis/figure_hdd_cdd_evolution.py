from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from  plotly  import  tools

import os
from plotly.offline import plot
import pandas as pd
from configuration import CONFIG_FILE, DATA_HDD_CDD_PLOTS_FOLDER, DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE, DATA_RAW_BUILDING_TODAY_HDD_FOLDER



def calc_graph(data_frame, analysis_fields, output_path, field_names, colors, i, fig):

    # CALCULATE GRAPH
    traces_graph = []
    for field, name, color in zip(analysis_fields, field_names, colors):
        x = data_frame.index
        y = data_frame[field]
        trace = go.Bar(x=x, y=y, name=name, orientation='v',
                       marker=dict(color=color))

        fig.append_trace(trace, 1, i + 1)

        # # PLOT GRAPH
        # layout = go.Layout(barmode='relative')
        # fig = go.Figure(data=traces_graph, layout=layout)
        # plot(fig, auto_open=False, filename=output_path + '//' + str(year) + ".html")

    return fig

def main(years_to_map, output_path, future_hdd_cdd_file, analysis_fields, field_names, colors, cities):
    # get data from every city and transform into data per scenario
    scenarios = future_hdd_cdd_file["Scenario"].unique()
    fig = tools.make_subplots(rows=1, cols=len(cities), shared_yaxes=True)

    for i, city in enumerate(cities):
        data_current = pd.read_csv(os.path.join(DATA_RAW_BUILDING_TODAY_HDD_FOLDER, city + ".csv")).set_index(
            "site_year")
        HDD_baseline = data_current.loc[2010, "hdd_18.5C"]
        CDD_baseline = data_current.loc[2010, "cdd_18.5C"]

        data_future_HDD_CDD = future_hdd_cdd_file[future_hdd_cdd_file['City'] == city]
        # HDD_baseline = data_future_HDD_CDD.loc["A1B_2020", "HDD_18_5_C"]
        # CDD_baseline = data_future_HDD_CDD.loc["A1B_2020", "CDD_18_5_C"]
        df_super_final = pd.DataFrame()
        for year in years_to_map:
            # HEATING CASE
            data_future_HDD_CDD["YEAR"] = [x.split("_", 1)[1] for x in data_future_HDD_CDD["Scenario"].values]
            data_future_HDD_CDD[year] = data_future_HDD_CDD["HDD_18_5_C"]
            data_future_HDD_CDD["SCENARIO_CLASS"] = [x.split("_", 1)[0] for x in data_future_HDD_CDD["Scenario"].values]
            data_future_HDD_CDD.set_index("SCENARIO_CLASS", inplace =True)
            df_heating = data_future_HDD_CDD[data_future_HDD_CDD.YEAR == str(year)]
            df_heating = df_heating[[year]]
            df_heating = df_heating.T

            # scenarios_extremes = ["A1B", "A2"]
            # for scenario in scenarios_extremes:
            #     df_heating[scenario] = df_heating[scenario] - df_heating["B1"]

            # COOLING CASE
            data_future_HDD_CDD[year] = -data_future_HDD_CDD["CDD_18_5_C"]
            data_future_HDD_CDD["SCENARIO_CLASS"] = [x.split("_", 1)[0]+"_CDD" for x in data_future_HDD_CDD["Scenario"].values]
            data_future_HDD_CDD.set_index("SCENARIO_CLASS", inplace=True)
            df_cooling = data_future_HDD_CDD[data_future_HDD_CDD.YEAR == str(year)]
            df_cooling = df_cooling[[year]]
            df_cooling = df_cooling.T

            # scenarios_extremes = ["A1B_CDD", "A2_CDD"]
            # for scenario in scenarios_extremes:
            #     df_cooling[scenario] = df_cooling[scenario] - df_cooling["B1_CDD"]

            df_final = pd.concat([df_heating, df_cooling], axis=1)
            df_super_final = df_super_final.append(df_final)

        fig = calc_graph(df_super_final, analysis_fields, output_path, field_names, colors, i, fig)

    fig['layout'].update(height=400, width=600, title='Multiple Subplots with Shared Y-Axes', barmode='relative',
                         )
    plot(fig, auto_open=False, filename=output_path + '//' + str(year) + ".html")




if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'IPCC_scenarios', 'data')
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
    output_path = DATA_HDD_CDD_PLOTS_FOLDER
    future_hdd_cdd_file = pd.read_csv(DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE)
    analysis_fields = ["A1B", "A1B_CDD"] #["B1", "A1B", "A2", "B1_CDD", "A1B_CDD", "A2_CDD"]
    field_names = ["Change in HDD [%]", "Change in CDD [%]"]
    colors = ["rgb(240,75,91)", "rgb(63,192,194)"] #["rgb(240,75,91)","rgb(246,148,143)","rgb(252,217,210)", "rgb(63,192,194)", "rgb(171,221,222)", "rgb(225,242,242)"]
    years_to_map = [2020, 2030, 2040, 2050]

    main(years_to_map, output_path, future_hdd_cdd_file, analysis_fields, field_names, colors, cities)