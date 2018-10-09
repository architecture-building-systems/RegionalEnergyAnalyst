import os

import numpy as np
import pandas as pd

from configuration import DATA_RAW_BUILDING_PERFORMANCE_FOLDER, DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER, \
    CONFIG_FILE, DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE, DATA_FUTURE_EFFICIENCY_FILE, DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE_EFFICEINCY


def main(cities, data_ipcc_folder, outputpath_file, scenarios, temperatures_base_H_C, temperatures_base_C_C,
         relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C, flag_use_efficiency):
    final_df = calc_enthalpy(cities, data_ipcc_folder, scenarios, temperatures_base_H_C, temperatures_base_C_C,
                             relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C, flag_use_efficiency)
    if flag_use_efficiency:
        outputpath_file = DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE_EFFICEINCY
    final_df.to_csv(outputpath_file, index=False)
    return


def calc_enthalpy(cities, data_ipcc_folder, scenarios, temperatures_base_H_C, temperatures_base_C_C,
                  relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C, flag_use_efficiency = False):
    final_df = pd.DataFrame()
    data_efficiency = pd.read_excel(DATA_FUTURE_EFFICIENCY_FILE, sheet_name="data").set_index('year')
    name_of_data_file = [x.split(",")[0] + "_" + x.split(", ")[-1] + "-hour.dat" for x in cities]
    name_of_data_file = [x.replace(" ", "_") for x in name_of_data_file]
    for scenario in scenarios:
        if flag_use_efficiency == True:
            year_scenario = scenario.split("data_")[1]
            if year_scenario == "1990_2010": #aka today
                today_efficiency = data_efficiency.loc[2010]
                temperatures_base_H_C = today_efficiency["Tb_H_A1B"]
                temperatures_base_C_C = today_efficiency["Tb_C_A1B"]
                relative_humidity_base_HUM_C = today_efficiency["RH_HUM_A1B"]
                relative_humidity_base_DEHUM_C = today_efficiency["RH_DEHUM_A1B"]
            else:
                year_real_scenario = scenario.split("_")[-1]
                today_efficiency = data_efficiency.loc[int(year_real_scenario)]
                scenario_type = scenario.split("_")[1]
                temperatures_base_H_C = today_efficiency["Tb_H_"+scenario_type]
                temperatures_base_C_C = today_efficiency["Tb_C_"+scenario_type]
                relative_humidity_base_HUM_C = today_efficiency["RH_HUM_"+scenario_type]
                relative_humidity_base_DEHUM_C = today_efficiency["RH_DEHUM_"+scenario_type]

        for name_file, city in zip(name_of_data_file, cities):
            weather_file_location = os.path.join(data_ipcc_folder, scenario, name_file)

            # Quantities
            weather_file = pd.read_csv(weather_file_location, sep='\s+', header=2, skiprows=0)
            temperatures_out_C = weather_file["Ta"].values[:8760]
            relative_humidity_percent = weather_file["RH"].values[:8760]
            humidity_ratio_out_kgperkg = np.vectorize(convert_rh_to_moisture_content)(relative_humidity_percent,
                                                                                      temperatures_out_C)
            humidity_ratio_base_H_kgperkg = convert_rh_to_moisture_content(relative_humidity_base_HUM_C, temperatures_base_H_C)
            humidity_ratio_base_C_kgperkg = convert_rh_to_moisture_content(relative_humidity_base_DEHUM_C, temperatures_base_C_C)

            deltah_kJ_kg, deltah_H_SEN_kJ_kg, deltah_C_SEN_kJ_kg, \
            deltah_HUM_kJ_kg, deltah_DEHUM_kJ_kg = np.vectorize(calc_yearly_enthalpy)(temperatures_out_C,
                                                                        humidity_ratio_out_kgperkg,
                                                                        temperatures_base_H_C,
                                                                        temperatures_base_C_C,
                                                                        humidity_ratio_base_H_kgperkg,
                                                                        humidity_ratio_base_C_kgperkg)

            aggregated_delta_enthalpy_H_SEN_KJ_kg = sum(deltah_H_SEN_kJ_kg) / 24
            aggregated_delta_enthalpy_C_SEN_KJ_kg = sum(deltah_C_SEN_kJ_kg) / 24
            aggregated_delta_enthalpy_HUM_KJ_kg = sum(deltah_HUM_kJ_kg) / 24
            aggregated_delta_enthalpy_DEHUM_KJ_kg = sum(deltah_DEHUM_kJ_kg) / 24

            HKKD_KJ_kg = sum(deltah_kJ_kg) / 24
            HKKD_SEN_KJ_kg = aggregated_delta_enthalpy_H_SEN_KJ_kg + aggregated_delta_enthalpy_C_SEN_KJ_kg
            SHR = HKKD_SEN_KJ_kg / HKKD_KJ_kg

            # list of fields to extract
            fields = {"City": [city],
                      "Scenario": scenario.split("_", 1)[1:],
                      "year": [scenario.split("_")[-1]],
                      "HKKD_H_sen_kJ_kg": [aggregated_delta_enthalpy_H_SEN_KJ_kg],
                      "HKKD_C_sen_kJ_kg": [aggregated_delta_enthalpy_C_SEN_KJ_kg],
                      "HKKD_HUM_kJ_kg": [aggregated_delta_enthalpy_HUM_KJ_kg],
                      "HKKD_DEHUM_kJ_kg": [aggregated_delta_enthalpy_DEHUM_KJ_kg],
                      "HKKD_kJ_kg": [HKKD_KJ_kg],
                      "SHR": [SHR]
                      }
            final_df = pd.concat([final_df, pd.DataFrame(fields)], ignore_index=True)
            print("scenario and city done: ", scenario, city)
    return final_df


def calc_h_lat(t_dry_C, humidity_ratio_out_kgperkg):
    """
    Calc specific temperature of moist air (latent)

    :param temperatures_out_C:
    :param CPA:
    :return:
    """
    CPW_kJ_kgC = 1.84
    h_we_kJ_kg = 2501

    h_kJ_kg = humidity_ratio_out_kgperkg * (t_dry_C * CPW_kJ_kgC + h_we_kJ_kg)

    return h_kJ_kg


def calc_h_sen(t_dry_C):
    """
    Calc specific temperature of moist air (sensible)

    :param temperatures_out_C:
    :param CPA:
    :return:
    """
    CPA_kJ_kgC = 1.006
    h_kJ_kg = t_dry_C * CPA_kJ_kgC

    return h_kJ_kg


def calc_yearly_enthalpy(temperatures_out_C, humidity_ratio_out_kgperkg, temperatures_base_H_C, temperatures_base_C_C,
                         humidity_ratio_base_H_kgperkg, humidity_ratio_base_C_kgperk):
    # calculate indoor and outdoor enthalpy for each state
    h_outdoor_lat_kJ_kg = calc_h_lat(temperatures_out_C, humidity_ratio_out_kgperkg)
    h_outdoor_sen_kJ_kg = calc_h_sen(temperatures_out_C)

    h_indoor_sen_H_kJ_kg = calc_h_sen(temperatures_base_H_C)
    h_indoor_sen_C_kJ_kg = calc_h_sen(temperatures_base_C_C)

    h_indoor_kJ_HUM_kg = calc_h_lat(temperatures_base_H_C, humidity_ratio_base_H_kgperkg)
    h_indoor_kJ_DEHUM_kg = calc_h_lat(temperatures_base_C_C, humidity_ratio_base_C_kgperk)

    # calculate provisional deltas
    deltah_H_SEN_kJ_kg = h_outdoor_sen_kJ_kg - h_indoor_sen_H_kJ_kg
    deltah_C_SEN_kJ_kg = h_outdoor_sen_kJ_kg - h_indoor_sen_C_kJ_kg

    deltah_HUM_kJ_kg = h_outdoor_lat_kJ_kg - h_indoor_kJ_HUM_kg
    deltah_DEHUM_kJ_kg = h_outdoor_lat_kJ_kg - h_indoor_kJ_DEHUM_kg

    # condition to see if there is sensible heating need
    if deltah_H_SEN_kJ_kg > 0:
        deltah_H_SEN_kJ_kg = 0.0
    else:
        deltah_H_SEN_kJ_kg = abs(deltah_H_SEN_kJ_kg)

    # condition to see if there is sensible cooling need
    if deltah_C_SEN_kJ_kg > 0:
        deltah_C_SEN_kJ_kg = abs(deltah_C_SEN_kJ_kg)
    else:
        deltah_C_SEN_kJ_kg = 0.0

    # condition to see if there is humidification
    if deltah_HUM_kJ_kg > 0:
        deltah_HUM_kJ_kg = 0.0
    else:
        deltah_HUM_kJ_kg = abs(deltah_HUM_kJ_kg)

    # condition to see if there is dehumidification
    if deltah_DEHUM_kJ_kg > 0:
        deltah_DEHUM_kJ_kg = abs(deltah_DEHUM_kJ_kg)
    else:
        deltah_DEHUM_kJ_kg = 0.0

    deltah_kJ_kg = deltah_H_SEN_kJ_kg + deltah_C_SEN_kJ_kg + deltah_HUM_kJ_kg + deltah_DEHUM_kJ_kg

    return deltah_kJ_kg, deltah_H_SEN_kJ_kg, deltah_C_SEN_kJ_kg, deltah_HUM_kJ_kg, deltah_DEHUM_kJ_kg


def convert_rh_to_moisture_content(rh, theta):
    """
    convert relative humidity to moisture content
    """
    import math
    P_ATM = 101325  # (Pa) atmospheric pressure
    p_sat_int = 611.2 * math.exp(17.62 * theta / (243.12 + theta))
    x = 0.622 * rh / 100 * p_sat_int / P_ATM
    return x


if __name__ == "__main__":

    y_field = "LOG_SITE_ENERGY_MWh_yr"
    data_energy_folder = DATA_RAW_BUILDING_PERFORMANCE_FOLDER
    data_ipcc_folder = DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER
    outputpath_file = DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE
    scenarios = ["data_1990_2010", "data_A1B_2010", "data_A1B_2020", "data_A1B_2030", "data_A1B_2040", "data_A1B_2050",
                 "data_A1B_2060", "data_A1B_2070", "data_A1B_2080", "data_A1B_2090", "data_A1B_2100",
                 "data_A2_2010", "data_A2_2020", "data_A2_2030", "data_A2_2040", "data_A2_2050", "data_A2_2060",
                 "data_A2_2070", "data_A2_2080", "data_A2_2090", "data_A2_2100",
                 "data_B1_2010", "data_B1_2020", "data_B1_2030", "data_B1_2040", "data_B1_2050", "data_B1_2060",
                 "data_B1_2070", "data_B1_2080", "data_B1_2090", "data_B1_2100"]
    data_path = os.path.abspath(os.path.dirname(__file__))
    data_efficiency = pd.read_excel(DATA_FUTURE_EFFICIENCY_FILE, sheet_name="data").set_index('year')
    today_efficiency = data_efficiency.loc[2010]
    temperatures_base_H_C = today_efficiency["Tb_H_A1B"]
    temperatures_base_C_C = today_efficiency["Tb_C_A1B"]
    relative_humidity_base_HUM_C = today_efficiency["RH_HUM_A1B"]
    relative_humidity_base_DEHUM_C = today_efficiency["RH_DEHUM_A1B"]

    #ACTIVATE FLAG USE EFFICEICNY TO SEE HOW BUILDING EFFICEINCY AFFECT THE THERMAL NEEDS
    flag_use_efficiency = False

    cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')['City'].values
    main(cities, data_ipcc_folder, outputpath_file, scenarios, temperatures_base_H_C, temperatures_base_C_C,
         relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C, flag_use_efficiency)
