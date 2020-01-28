import os

import numpy as np
import pandas as pd

from configuration import DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY, DATA_PREDICTION_FOLDER_FUTURE_EFFICIENCY, ZONE_NAMES, \
    CONFIG_FILE, DATA_RAW_BUILDING_PERFORMANCE_FOLDER, DATA_RAW_BUILDING_TODAY_HDD_FOLDER,\
    DATA_FUTURE_EFFICIENCY_FILE, DATA_ALLDATA_FILE,DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER
from data_processing.enthalpy_calculation import convert_rh_to_moisture_content, calc_yearly_enthalpy
from data_processing.training_and_testing_database import calc_massflow_building, calc_total_energy

data_path = os.path.abspath(os.path.dirname(__file__))
data_efficiency = pd.read_excel(DATA_FUTURE_EFFICIENCY_FILE, sheet_name="data").set_index('year')


def main(cities, climate, scenarios, data_energy_folder, data_ipcc_folder, output_path):
    name_of_data_file = [x.split(",")[0] + "_" + x.split(", ")[-1] + "-hour.dat" for x in cities]
    name_of_data_file = [x.replace(" ", "_") for x in name_of_data_file]

    # get climate calssification
    new_clima = []
    for clima in climate:
        for category, categories in ZONE_NAMES.items():
            if clima.split(" ")[0] in categories:
                new_clima.append(category)

    # get training and testing dataset
    data_train_test_all = pd.read_csv(DATA_ALLDATA_FILE)

    for name_file, city, climate in zip(name_of_data_file, cities, new_clima):
        final_df = pd.DataFrame()
        for scenario in scenarios:
            output_path = DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY
            today_efficiency = data_efficiency.loc[2010]
            temperatures_base_H_C = today_efficiency["Tb_H_A1B"]
            temperatures_base_C_C = today_efficiency["Tb_C_A1B"]
            relative_humidity_base_HUM_C = today_efficiency["RH_HUM_A1B"]
            relative_humidity_base_DEHUM_C = today_efficiency["RH_DEHUM_A1B"]
            COP_H = today_efficiency["COP_H_A1B"]
            COP_C = today_efficiency["COP_C_A1B"]

            # get data for city NEEDED TO KEEP THE INDEX!!! DO NOT ERASE
            data_measured = pd.read_csv(os.path.join(data_energy_folder, city + ".csv"))
            data_measured["site_year"] = data_measured["site_year"].round(0)
            df_hdd_cdd = pd.read_csv(os.path.join(DATA_RAW_BUILDING_TODAY_HDD_FOLDER, city + ".csv"))
            df_hdd_cdd["site_year"] = df_hdd_cdd["site_year"].round(0)
            data = pd.merge(df_hdd_cdd, data_measured, on="site_year")
            data["BUILDING_ID"] = [city + str(ix) for ix in data.index]

            # merge to data about the cluster type
            data_train_test_city = data_train_test_all[['BUILDING_ID', 'CLUSTER_LOG_SITE_EUI_kWh_m2yr']]
            data = pd.merge(data, data_train_test_city, on="BUILDING_ID")

            # get other quantities
            data['CITY'] = city
            data['CLIMATE_ZONE'] = climate
            data['SCENARIO'] = scenario.split("data_")[1]

            # Quantities
            weather_file_location = os.path.join(data_ipcc_folder, scenario, name_file)
            weather_file = pd.read_csv(weather_file_location, sep='\s+', header=2, skiprows=0)
            temperatures_out_C = weather_file["Ta"].values[:8760]
            relative_humidity_percent = weather_file["RH"].values[:8760]
            humidity_ratio_out_kgperkg = np.vectorize(convert_rh_to_moisture_content)(relative_humidity_percent,
                                                                                      temperatures_out_C)

            humidity_ratio_base_H_kgperkg = convert_rh_to_moisture_content(relative_humidity_base_HUM_C,
                                                                           temperatures_base_H_C)
            humidity_ratio_base_C_kgperkg = convert_rh_to_moisture_content(relative_humidity_base_DEHUM_C,
                                                                           temperatures_base_C_C)

            delta_enthalpy_kJ_kg, _, _, _, _ = np.vectorize(calc_yearly_enthalpy)(temperatures_out_C,
                                                                                  humidity_ratio_out_kgperkg,
                                                                                  temperatures_base_H_C,
                                                                                  temperatures_base_C_C,
                                                                                  humidity_ratio_base_H_kgperkg,
                                                                                  humidity_ratio_base_C_kgperkg)

            # Quantities of every building
            data["GROSS_FLOOR_AREA_m2"] = (data["floor_area"] * 0.092903).values
            data["BUILDING_CLASS"] = data["building_class"].values
            volumetric_flow_building_m3_s = np.vectorize(calc_massflow_building)(data["GROSS_FLOOR_AREA_m2"].values,
                                                                                 data["BUILDING_CLASS"].values)

            data["THERMAL_ENERGY_kWh_yr"] = [calc_total_energy(x, delta_enthalpy_kJ_kg, y, COP_H, COP_C) for x, y in
                                             zip(volumetric_flow_building_m3_s, data["BUILDING_CLASS"].values)]

            # logarithmic values
            data['LOG_THERMAL_ENERGY_kWh_yr'] = np.log(data["THERMAL_ENERGY_kWh_yr"].values)

            # list of fields to extract
            fields = ["BUILDING_ID",
                      "CITY",
                      "CLIMATE_ZONE",
                      "SCENARIO",
                      "BUILDING_CLASS",
                      "GROSS_FLOOR_AREA_m2",
                      "LOG_THERMAL_ENERGY_kWh_yr",
                      "CLUSTER_LOG_SITE_EUI_kWh_m2yr"
                      ]

            final_df = pd.concat([final_df, data[fields]], ignore_index=True)
        final_df.to_csv(os.path.join(output_path, city + ".csv"), index=False)
        print("city done: ", city)


if __name__ == "__main__":
    scenarios = ["data_1990_2010", "data_A1B_2010", "data_A1B_2020", "data_A1B_2030", "data_A1B_2040", "data_A1B_2050",
                 "data_A1B_2060", "data_A1B_2070", "data_A1B_2080", "data_A1B_2090", "data_A1B_2100",
                 "data_A2_2010", "data_A2_2020", "data_A2_2030", "data_A2_2040", "data_A2_2050", "data_A2_2060",
                 "data_A2_2070", "data_A2_2080", "data_A2_2090", "data_A2_2100",
                 "data_B1_2010", "data_B1_2020", "data_B1_2030", "data_B1_2040", "data_B1_2050", "data_B1_2060",
                 "data_B1_2070", "data_B1_2080", "data_B1_2090", "data_B1_2100"]
    flag_use_efficiency = False
    data_energy_folder = DATA_RAW_BUILDING_PERFORMANCE_FOLDER
    data_ipcc_folder = DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER
    output_path = DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY
    cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')['City'].values
    climate = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')['climate'].values
    main(cities, climate, scenarios, data_energy_folder, data_ipcc_folder, output_path)
