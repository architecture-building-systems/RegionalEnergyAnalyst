import pandas as pd
from configuration import CONFIG_FILE, NN_MODEL_PREDICTION_FOLDER, DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, DATA_ALLDATA_FILE,\
    DATA_POST_FUTURE_ENERGY_CONSUMPTION_FILE_WITH_EFFICIENCY,DATA_POST_FUTURE_ENERGY_CONSUMPTION_FILE, DATA_PREDICTION_FOLDER_FUTURE_EFFICIENCY, DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY
import numpy as np
import os

model = "log_nn_wd_4L_2var"
data_cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')
test_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
cities = data_cities['City'].values
latitudes = data_cities['latitude'].values
longitudes = data_cities['longitude'].values
climates = data_cities['climate'].values

flag_use_efficiency = False
if flag_use_efficiency == True:
    output_final = DATA_POST_FUTURE_ENERGY_CONSUMPTION_FILE_WITH_EFFICIENCY
    data_model_folder = os.path.join(NN_MODEL_PREDICTION_FOLDER, model, "future_efficiency")
else:
    output_final = DATA_POST_FUTURE_ENERGY_CONSUMPTION_FILE
    data_model_folder = os.path.join(NN_MODEL_PREDICTION_FOLDER, model, "today_efficiency")

scenarios = ["data_1990_2010", "data_A1B_2010", "data_A1B_2020", "data_A1B_2030", "data_A1B_2040", "data_A1B_2050",
             "data_A1B_2060", "data_A1B_2070", "data_A1B_2080", "data_A1B_2090", "data_A1B_2100",
             "data_A2_2010", "data_A2_2020", "data_A2_2030", "data_A2_2040", "data_A2_2050", "data_A2_2060",
             "data_A2_2070", "data_A2_2080", "data_A2_2090", "data_A2_2100",
             "data_B1_2010", "data_B1_2020", "data_B1_2030", "data_B1_2040", "data_B1_2050", "data_B1_2060",
             "data_B1_2070", "data_B1_2080", "data_B1_2090", "data_B1_2100"]

scenarios_today = ["data_1990_2010", "data_A2_2010", "data_B1_2010", "data_A1B_2010"]
final_df = pd.DataFrame()
for scenario1 in scenarios:
    scenario = scenario1.split("_",1)[1:][0]
    list_of_dicts = []
    for city, latitude, longitude, climate  in zip(cities, latitudes, longitudes, climates):
        print(city)
        if scenario in scenarios_today:
            all_data = pd.read_csv(DATA_ALLDATA_FILE)
            data2 = all_data[all_data["SCENARIO"] == scenario1]
            data = all_data[all_data["CITY"] == city]
        else:
            all_data = pd.read_csv(os.path.join(data_model_folder, city+".csv"))
            data = all_data[all_data["SCENARIO"] == scenario]
        energy_MWh = np.mean(data["SITE_ENERGY_kWh_yr"].values/1000).round(2)
        EUI_kWh_m2yr = np.mean(data["SITE_ENERGY_kWh_yr"].values / data["GROSS_FLOOR_AREA_m2"].values).round(2)
        std_EUI_kWh_m2yr = np.std(EUI_kWh_m2yr).round(2)
        std_Energy_MWh_yr = np.std(energy_MWh).round(2)
        if city in test_cities:
            label = 125.50
        else:
            label = 225.50
        dictionary = {"1_city": city,
             "2_climate_zone": climate.split(" ")[0],
             "2.1_climate_description": climate.split(" (")[:-1][0].split(" ", 1)[-1:][0],
             "energy_MWh": energy_MWh,
             "EUI_kWh_m2yr": EUI_kWh_m2yr,
             "scenario":scenario,
             "std_Energy_MWh_yr": std_Energy_MWh_yr,
             "8_m_label": label,
             "9_lat": latitude,
             "10_lon": longitude,
             }
        list_of_dicts.append(dictionary)
    df_output = pd.DataFrame(list_of_dicts)
    final_df = pd.concat([final_df, df_output], ignore_index=True)

    if scenario =="1990_2010":
        df_baseline = df_output
    else:
        df_output["delta_energy"] = (df_output["energy_MWh"] - df_baseline["energy_MWh"]).round(2)
        df_output["per_energy"] = (df_output["delta_energy"] /df_baseline["energy_MWh"]*100).round(2)
        df_output["delta_eui"] = (df_output["EUI_kWh_m2yr"] - df_baseline["EUI_kWh_m2yr"]).round(2)
        df_output["per_eui"] = (df_output["delta_eui"] /df_baseline["EUI_kWh_m2yr"]*100).round(2)

    # output = os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_consumption" + scenario + ".csv")
    # df_output.to_csv(output, sep=";", index=False)

final_df.to_csv(output_final)
