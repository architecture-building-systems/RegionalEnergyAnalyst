from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
import os
from plotly.offline import plot
import pandas as pd
from configuration import CONFIG_FILE, DATA_HDD_CDD_PLOTS_FOLDER, DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE, DATA_RAW_BUILDING_TODAY_HDD_FOLDER


def main(data_path, output_path, future_hdd_cdd_file, cities):
    # get data from every city and transform into data per scenario
    scenarios = future_hdd_cdd_file["Scenario"].unique()
    data_table = pd.DataFrame()
    for city in cities:
        data_current = pd.read_csv(os.path.join(DATA_RAW_BUILDING_TODAY_HDD_FOLDER, city + ".csv")).set_index("site_year")
        data_new = future_hdd_cdd_file[future_hdd_cdd_file['City'] == city]
        data_new2 = data_new.set_index("Scenario")

        # HEATING CASE
        HDD_18_5_C = data_current.loc[2010, "hdd_18.5C"]
        CDD_18_5_C = data_current.loc[2010, "cdd_18.5C"]
        data_new["YEAR"] = [x.split("_", 1)[1] for x in data_new["Scenario"].values]
        data_new.set_index("YEAR", inplace=True)
        data_new["CHANGE"] = ((data_new["HDD_18_5_C"] - HDD_18_5_C) / HDD_18_5_C) * 100
        data_new["SCENARIO_CLASS"] = [x.split("_", 1)[0] for x in data_new["Scenario"].values]
        data_final = pd.DataFrame()
        for scenario in scenarios:
            scenario_type = scenario.split("_", 1)[1]
            df = data_new[data_new.index == scenario_type]
            # df[scenario_type] =
            data_final[scenario_type] = pd.DataFrame({scenario_type: df["HDD_18_5_C"].values})

        fig, ax = plt.subplots()
        data_final[['2020', '2030', '2040', '2050']].plot.box(title=city, figsize=(4, 4), ax=ax)

        # COOLING CASE
        data_new["YEAR"] = [x.split("_", 1)[1] for x in data_new["Scenario"].values]
        data_new.set_index("YEAR", inplace=True)
        data_new["CHANGE"] = ((data_new["CDD_18_5_C"] - CDD_18_5_C) / CDD_18_5_C) * 100
        data_new["SCENARIO_CLASS"] = [x.split("_", 1)[0] for x in data_new["Scenario"].values]
        data_final = pd.DataFrame()
        for scenario in scenarios:
            scenario_type = scenario.split("_", 1)[1]
            df = data_new[data_new.index == scenario_type]
            # df[scenario_type] =
            data_final[scenario_type] = pd.DataFrame({scenario_type: df["CDD_18_5_C"].values})

        data_final[['2020', '2030', '2040', '2050']].plot.box(title=city, figsize=(4, 4), ax=ax)
        plt.ylim((0, 3500))
        plt.savefig(os.path.join(output_path, city + ".png"))
        #
        #     data_final['scenario'] = df["SCENARIO_CLASS"].values
        #     data_final['city'] = city
        #
        #     data_table = data_table.append(data_final)
        # data_table.to_csv(os.path.join(output_path, cooling_heating, cooling_heating+".csv"))


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'IPCC_scenarios', 'data')
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
    output_path = DATA_HDD_CDD_PLOTS_FOLDER
    future_hdd_cdd_file = pd.read_csv(DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE)

    main(data_path, output_path, future_hdd_cdd_file, cities)




cities_energy_data = pd.read_excel(CONFIG_FILE, sheet_name="cities_with_energy_data")
data_hdd = pd.read_csv(DATA_RAW_BUILDING_TODAY_HDD_FOLDER)

trace1 = go.Bar(
    y=['giraffes', 'orangutans', 'monkeys'],
    x=[20, 14, 23],
    name='SF Zoo',
    orientation = 'h',
    marker = dict(
        color = 'rgba(246, 78, 139, 0.6)',
        line = dict(
            color = 'rgba(246, 78, 139, 1.0)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=['giraffes', 'orangutans', 'monkeys'],
    x=[12, 18, 29],
    name='LA Zoo',
    orientation = 'h',
    marker = dict(
        color = 'rgba(58, 71, 80, 0.6)',
        line = dict(
            color = 'rgba(58, 71, 80, 1.0)',
            width = 3)
    )
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='marker-h-bar')