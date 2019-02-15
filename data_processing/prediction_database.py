import os

import numpy as np
import pandas as pd

from configuration import DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY, DATA_PREDICTION_FOLDER_FUTURE_EFFICIENCY, ZONE_NAMES, \
    CONFIG_FILE, DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER, DATA_RAW_BUILDING_PERFORMANCE_FOLDER, \
    DATA_FUTURE_EFFICIENCY_FILE
from data_processing.enthalpy_calculation import convert_rh_to_moisture_content, calc_yearly_enthalpy
from data_processing.training_and_testing_database import calc_massflow_building, calc_total_energy

data_path = os.path.abspath(os.path.dirname(__file__))
data_efficiency = pd.read_excel(DATA_FUTURE_EFFICIENCY_FILE, sheet_name="data").set_index('year')


def main(cities, climate, scenarios):
    name_of_data_file = [x.split(",")[0] + "_" + x.split(", ")[-1] + "-hour.dat" for x in cities]
    name_of_data_file = [x.replace(" ", "_") for x in name_of_data_file]

    # get climate calssification
    new_clima = []
    for clima in climate:
        for category, categories in ZONE_NAMES.items():
            if clima.split(" ")[0] in categories:
                new_clima.append(category)

    for name_file, city, climate in zip(name_of_data_file, cities, new_clima):
        final_df = pd.DataFrame()
        for scenario in scenarios:
            if flag_use_efficiency == True:
                output_path = DATA_PREDICTION_FOLDER_FUTURE_EFFICIENCY
                year_scenario = scenario.split("data_")[1]
                if year_scenario == "1990_2010":  # aka today
                    today_efficiency = data_efficiency.loc[2010]
                    temperatures_base_H_C = today_efficiency["Tb_H_A1B"]
                    temperatures_base_C_C = today_efficiency["Tb_C_A1B"]
                    relative_humidity_base_HUM_C = today_efficiency["RH_HUM_A1B"]
                    relative_humidity_base_DEHUM_C = today_efficiency["RH_DEHUM_A1B"]
                    COP_H = today_efficiency["COP_H_A1B"]
                    COP_C = today_efficiency["COP_C_A1B"]
                else:
                    year_real_scenario = scenario.split("_")[-1]
                    today_efficiency = data_efficiency.loc[int(year_real_scenario)]
                    scenario_type = scenario.split("_")[1]
                    temperatures_base_H_C = today_efficiency["Tb_H_" + scenario_type]
                    temperatures_base_C_C = today_efficiency["Tb_C_" + scenario_type]
                    relative_humidity_base_HUM_C = today_efficiency["RH_HUM_" + scenario_type]
                    relative_humidity_base_DEHUM_C = today_efficiency["RH_DEHUM_" + scenario_type]
                    COP_H = today_efficiency["COP_H_" + scenario_type]
                    COP_C = today_efficiency["COP_C_" + scenario_type]
            else:
                output_path = DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY
                today_efficiency = data_efficiency.loc[2010]
                temperatures_base_H_C = today_efficiency["Tb_H_A1B"]
                temperatures_base_C_C = today_efficiency["Tb_C_A1B"]
                relative_humidity_base_HUM_C = today_efficiency["RH_HUM_A1B"]
                relative_humidity_base_DEHUM_C = today_efficiency["RH_DEHUM_A1B"]
                COP_H = today_efficiency["COP_H_A1B"]
                COP_C = today_efficiency["COP_C_A1B"]

            # get enthalpy
            df_join = pd.read_csv(os.path.join(DATA_RAW_BUILDING_PERFORMANCE_FOLDER, city + ".csv"))
            weather_file_location = os.path.join(DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER, scenario, name_file)

            # Quantities
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
            gross_floor_area_m2 = (df_join["floor_area"] * 0.092903).values
            building_class = df_join["building_class"].values
            volumetric_flow_building_m3_s = np.vectorize(calc_massflow_building)(gross_floor_area_m2, building_class)

            thermal_energy_kWh_yr = [calc_total_energy(x, delta_enthalpy_kJ_kg, y, COP_H, COP_C) for x, y in
                                     zip(volumetric_flow_building_m3_s, building_class)]

            # list of fields to extract

            dataframe = pd.DataFrame({"THERMAL_ENERGY_kWh_yr": thermal_energy_kWh_yr,
                                      "LOG_THERMAL_ENERGY_kWh_yr": np.log(thermal_energy_kWh_yr)})
            dataframe["CITY"] = city
            dataframe["SCENARIO"] = scenario.split("_", 1)[1:][0]
            dataframe["YEAR"] = scenario.split("_")[-1]
            dataframe["BUILDING_CLASS"] = building_class
            dataframe["GROSS_FLOOR_AREA_m2"] = gross_floor_area_m2
            dataframe["CLIMATE_ZONE"] = climate
            final_df = pd.concat([final_df, dataframe], ignore_index=True)
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
    cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')['City'].values
    climate = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')['climate'].values
    main(cities, climate, scenarios)
