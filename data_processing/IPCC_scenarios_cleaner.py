import os

import numpy as np
import pandas as pd
import csv
from configuration import CONFIG_FILE, DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE, DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER

def main(scenarios, cities, data_path, output_path):

    name_of_data_file = [x.split(",")[0] +"_" + x.split(", ")[-1] +"-hour.dat" for x in cities]
    name_of_data_file = [x.replace(" ", "_") for x in name_of_data_file]
    final_df = pd.DataFrame()
    for scenario in scenarios:
        for name_file, city in zip(name_of_data_file, cities):
            weather_file_location = os.path.join(DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER, scenario, name_file)
            temperatures = pd.read_csv(weather_file_location, sep='\s+', header=2, skiprows=0)["Ta"].values
            HDD_18_5_C, CDD_18_5_C = cal_heating_degree_days(temperatures, 18.5)
            city_data = pd.DataFrame({"City": city,
                                      "Scenario": scenario.split("_", 1)[1:],
                                      "HDD_18_5_C": HDD_18_5_C,
                                      "CDD_18_5_C": CDD_18_5_C,})
            final_df = pd.concat([final_df, city_data], ignore_index=True)
    final_df.to_csv(output_path, index=False)


def cal_heating_degree_days(temperatures, base_temp):

    hdd = 0
    cdd = 0
    n =  24 # 24 hours
    temperatures_per_day = [temperatures[i:i + n] for i in range(0, len(temperatures), n)]
    for temp in temperatures_per_day:
        hdh = 0
        cdh = 0
        for t in temp:
            if t < base_temp:
                hdh += (1/24)*(base_temp-t)
            elif t > base_temp:
                cdh += (1 / 24) * (t - base_temp)
        hdd += hdh
        cdd += cdh

    return round(hdd,0), round(cdd,0)


def dat_to_csv(city, data_path, data_stations, scenario, scenario_year):
    weather_file_path = os.path.join(data_path, scenario, data_stations.loc[city, "Weather station"] + "-" + scenario_year + ".dat")
    weather_file_path_out = os.path.join(data_path, scenario, data_stations.loc[city, "Weather station"] + "-" + scenario_year + ".csv")
    newLines = [['a', 'b', 'c', 'd', 'e', 'global radiation', 'diffuse radiation', 'radiation on inclined plane', 'direct normal radiation', 'temperature']]
    with open(weather_file_path) as input_file:
        for line in input_file:
            newLine = [x.strip() for x in line.split('\t')]
            if len(newLine) == 10 and newLine[3] and newLine[4]:
                newLines.append(newLine)
    with open(weather_file_path_out, 'w') as output_file:
        file_writer = csv.writer(output_file)
        file_writer.writerows(newLines)


    return weather_file_path_out


if __name__ == "__main__":
    scenarios = ["data_1990_2010", "data_A1B_2010", "data_A1B_2020", "data_A1B_2030", "data_A1B_2040", "data_A1B_2050","data_A1B_2060", "data_A1B_2070", "data_A1B_2080", "data_A1B_2090", "data_A1B_2100",
                 "data_A2_2010", "data_A2_2020", "data_A2_2030", "data_A2_2040", "data_A2_2050", "data_A2_2060", "data_A2_2070", "data_A2_2080", "data_A2_2090", "data_A2_2100",
                 "data_B1_2010", "data_B1_2020", "data_B1_2030", "data_B1_2040", "data_B1_2050", "data_B1_2060", "data_B1_2070", "data_B1_2080", "data_B1_2090", "data_B1_2100"]
    data_path = os.path.abspath(os.path.dirname(__file__))
    output_path = DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE
    data_stations = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')["City"].values

    main(scenarios, data_stations, data_path, output_path)