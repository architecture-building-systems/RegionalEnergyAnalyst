import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix, parallel_coordinates


def main(data_path, output_path, cities):

    # get data from every city and transform into data per scenario
    scenarios = pd.read_csv(os.path.join(data_path, cities[0]+".csv"))["IPCC_SCENARIO"]
    for city in cities:
        data = pd.read_csv(os.path.join(data_path, city + ".csv"))
        data_new = data.set_index("IPCC_SCENARIO")
        today_value = data_new.loc["A1B_2020", "SITE_ENERGY_MWh_yr"]
        data["YEAR"] = [x.split("_",1)[1] for x in data["IPCC_SCENARIO"].values]
        data.set_index("YEAR", inplace=True)
        data["CHANGE"] = ((data["SITE_ENERGY_MWh_yr"] - today_value)/today_value)*100
        data["SCENARIO_CLASS"] = [x.split("_",1)[0] for x in data["IPCC_SCENARIO"].values]
        data_final = pd.DataFrame()
        for scenario in scenarios:
            scenario_type = scenario.split("_", 1)[1]
            df = data[data.index == scenario_type]
            # df[scenario_type] =
            data_final[scenario_type] = pd.DataFrame({scenario_type:df.CHANGE.values})

        data_final[['2030', '2040', '2050']].plot.box(title=city, figsize=(2.5,2.5))
        plt.ylim((-30, 30))
        plt.savefig(os.path.join(output_path, city+".png"))

if __name__ == "__main__":
    name_model = "log_neural_net_wide_deep_4L_453%_3%" #"log_log_all_standard_10000"
    data_path = os.path.join(os.path.dirname(os.getcwd()), "results", "predictions", name_model)
    output_path = os.path.join(os.path.dirname(os.getcwd()), "results", "predictions", name_model, "plots")
    cities = pd.read_excel(os.path.join(os.path.dirname(os.getcwd()), "cities.xlsx"), sheet_name='test_cities')['City']

    main(data_path, output_path, cities)


