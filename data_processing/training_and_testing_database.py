import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configuration import DATA_RAW_BUILDING_PERFORMANCE_FOLDER, CONFIG_FILE, DATA_TRAINING_FILE, \
    DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER, DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE, DATA_TESTING_FILE, \
    DATA_ALLDATA_FILE, DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE, DATA_RAW_BUILDING_TODAY_HDD_FOLDER, \
    DATA_FUTURE_EFFICIENCY_FILE
from data_processing.enthalpy_calculation import convert_rh_to_moisture_content, calc_yearly_enthalpy


def prepare_input_database(cities, data_energy_folder, delta_enthalpy_data, scenario, COP_H, COP_C,
                           temperatures_base_H_C, temperatures_base_C_C,
                           relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C):
    final_df = pd.DataFrame()
    name_of_data_file = [x.split(",")[0] + "_" + x.split(", ")[-1] + "-hour.dat" for x in cities]
    name_of_data_file = [x.replace(" ", "_") for x in name_of_data_file]
    data_1990_2010 = pd.read_csv(DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE)
    for name_file, city in zip(name_of_data_file, cities):
        # get_local_data:
        data_measured = pd.read_csv(os.path.join(data_energy_folder, city + ".csv"))
        data_measured["site_year"] = data_measured["site_year"].round(0)
        df_hdd_cdd = pd.read_csv(os.path.join(DATA_RAW_BUILDING_TODAY_HDD_FOLDER, city + ".csv"))
        df_hdd_cdd["site_year"] = df_hdd_cdd["site_year"].round(0)
        df_join = pd.merge(df_hdd_cdd, data_measured, on="site_year")

        # get enthalpy
        weather_file_location = os.path.join(DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER, scenario, name_file)

        # Quantities
        weather_file = pd.read_csv(weather_file_location, sep='\s+', header=2, skiprows=0)
        temperatures_out_C = weather_file["Ta"].values[:8760]
        relative_humidity_percent = weather_file["RH"].values[:8760]
        humidity_ratio_out_kgperkg = np.vectorize(convert_rh_to_moisture_content)(relative_humidity_percent,
                                                                                  temperatures_out_C)
        humidity_ratio_base_H_kgperkg = convert_rh_to_moisture_content(relative_humidity_base_HUM_C, temperatures_base_H_C)
        humidity_ratio_base_C_kgperkg = convert_rh_to_moisture_content(relative_humidity_base_DEHUM_C, temperatures_base_C_C)

        delta_enthalpy_kJ_kg, _, _, _, _ = np.vectorize(calc_yearly_enthalpy)(temperatures_out_C,
                                                                        humidity_ratio_out_kgperkg,
                                                                        temperatures_base_H_C,
                                                                        temperatures_base_C_C,
                                                                        humidity_ratio_base_H_kgperkg,
                                                                        humidity_ratio_base_C_kgperkg)

        # Quantities of every building
        df_join["SITE_ENERGY_kWh_yr"] = (df_join["site_energy"] * 0.293071).round(2)
        df_join["SITE_EUI_kWh_m2yr"] = (df_join["site_eui"] * 3.15459).round(2)
        gross_floor_area_m2 = (df_join["floor_area"] * 0.092903).values
        building_class = df_join["building_class"].values
        volumetric_flow_building_m3_s = np.vectorize(calc_massflow_building)(gross_floor_area_m2, building_class)

        # Standarize site energy consumption
        hhd_standard_year = data_1990_2010[data_1990_2010["City"] == city]
        hhd_standard_year = hhd_standard_year[hhd_standard_year["Scenario"] == "1990_2010"]
        hdd_cdd_standard = hhd_standard_year["HDD_18_5_C"].values[0] + hhd_standard_year["CDD_18_5_C"].values[0]
        site_energy_kWh_yr = (df_join["SITE_ENERGY_kWh_yr"] * (
                df_join["hdd_18.5C"] + df_join["cdd_18.5C"])) / hdd_cdd_standard
        site_EUI_kWh_m2yr = (df_join["SITE_EUI_kWh_m2yr"] * (
                df_join["hdd_18.5C"] + df_join["cdd_18.5C"])) / hdd_cdd_standard

        # Data about enthalpy days
        aggreagated_delta_enthalpy_standard_year = delta_enthalpy_data[delta_enthalpy_data["City"] == city]
        aggreagated_delta_enthalpy_standard_year = aggreagated_delta_enthalpy_standard_year[
            aggreagated_delta_enthalpy_standard_year["Scenario"] == "1990_2010"]

        HKKD_kJ_kg = aggreagated_delta_enthalpy_standard_year["HKKD_kJ_kg"].values[0]
        SHR = aggreagated_delta_enthalpy_standard_year["SHR"].values[0]

        # final calculation
        thermal_energy_kWh_yr = [calc_total_energy(x, delta_enthalpy_kJ_kg, y, COP_H, COP_C) for x, y in
                                 zip(volumetric_flow_building_m3_s, building_class)]

        # list of fields to extract
        fields = {"CITY": city,
                  "SCENARIO": scenario,
                  "YEAR": df_join["site_year"].values,
                  "BUILDING_CLASS": building_class,
                  "SITE_ENERGY_kWh_yr": site_energy_kWh_yr,
                  "SITE_EUI_kWh_m2yr": site_EUI_kWh_m2yr,
                  "GROSS_FLOOR_AREA_m2": gross_floor_area_m2,
                  "HKKD_kJ_kg": HKKD_kJ_kg,
                  "SHR": SHR,
                  "THERMAL_ENERGY_kWh_yr": thermal_energy_kWh_yr,
                  "LOG_SITE_ENERGY_kWh_yr": np.log(site_energy_kWh_yr),
                  "LOG_SITE_EUI_kWh_m2yr": np.log(site_EUI_kWh_m2yr),
                  "LOG_GROSS_FLOOR_AREA_m2": np.log(gross_floor_area_m2),
                  "LOG_THERMAL_ENERGY_kWh_yr": np.log(thermal_energy_kWh_yr),
                  "LOG_HKKD_kJ_kg": np.log(HKKD_kJ_kg),
                  "LOG_SHR": np.log(SHR),
                  }
        final_df = pd.concat([final_df, pd.DataFrame(fields)], ignore_index=True)
        print("scenario and city done: ", scenario, city)
    return final_df


def calc_power(delta_enthalpy_kJ_kg, hour_of_year, day_of_week, volumetric_flow_building_m3_s, building_class, COP_H,
               COP_C):
    density_air_kg_m3 = 1.202
    if delta_enthalpy_kJ_kg < 0:  # heating
        COP = COP_C
    else:
        COP = COP_H

    if building_class == "Residential":
        if day_of_week == 5:
            Occ_profile = [1, 1, 1, 1, 1, 1, 0.75, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7,
                           0.8, 0.9, 0.9]
        elif day_of_week == 6:
            Occ_profile = [1, 1, 1, 1, 1, 1, 0.75, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7,
                           0.8, 0.9, 0.9]
        else:
            Occ_profile = [1, 1, 1, 1, 1, 1, 0.75, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7,
                           0.8, 0.9, 0.9]
    elif building_class == "Commercial":
        if day_of_week == 5:
            Occ_profile = [0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.9, 0.9, 0.4, 0.4, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.1, 0.1,
                           0.1, 0, 0]
        elif day_of_week == 6:
            Occ_profile = [0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0]
        else:
            Occ_profile = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        Exception("errorrrr")

    Occ_ratio = Occ_profile[hour_of_year]
    power_kWh = abs(volumetric_flow_building_m3_s * density_air_kg_m3 * (delta_enthalpy_kJ_kg / COP)) * Occ_ratio

    return power_kWh


def calc_total_energy(volumetric_flow_building_m3_s, delta_enthalpy_kJ_kg, building_class, COP_H, COP_C):
    rng = pd.date_range('1/1/2011', periods=8760, freq='H')
    hour_of_year = rng.hour.values
    day_of_week = rng.dayofweek.values
    power_kWh_yr = np.vectorize(calc_power)(delta_enthalpy_kJ_kg, hour_of_year, day_of_week,
                                            volumetric_flow_building_m3_s, building_class, COP_H, COP_C)
    energy_MWh_yr = sum(power_kWh_yr)

    return energy_MWh_yr


def calc_massflow_building(gross_floor_area_m2, building_class):
    F = 3  # storey height
    if building_class == "Residential":
        ACH_1_h = 4
    elif building_class == "Commercial":
        ACH_1_h = 6
    else:
        Exception("errorrrr")

    volumetric_flow_building_m3_s = ACH_1_h * gross_floor_area_m2 * F / 3600

    return volumetric_flow_building_m3_s


def training_testing_splitter(final_df, cities, y_field, sizes):
    test_size = sizes[1]
    X_final_train = pd.DataFrame()
    X_final_test = pd.DataFrame()
    # split into training and test datasets
    for city in cities:
        data = final_df[final_df['CITY'] == city]
        y = data[y_field]
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size)

        # join each dataset
        X_final_train = pd.concat([X_final_train, X_train], ignore_index=True)
        X_final_test = pd.concat([X_final_test, X_test], ignore_index=True)

    return X_final_train, X_final_test


def main(cities, data_energy_folder, data_ipcc_file, outputpath_training, outputpath_testing, outputpath_all_data,
         y_field, sizes, scenarios, COP_H, COP_C, temperatures_base_H_C, temperatures_base_C_C,
         relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C):
    delta_enthalpy_data = pd.read_csv(data_ipcc_file)
    final_df = prepare_input_database(cities, data_energy_folder, delta_enthalpy_data, scenarios, COP_H, COP_C,
                                      temperatures_base_H_C, temperatures_base_C_C,
                                      relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C)

    training, testing = training_testing_splitter(final_df, cities, y_field, sizes)

    training.to_csv(outputpath_training, index=False)
    testing.to_csv(outputpath_testing, index=False)
    final_df.to_csv(outputpath_all_data, index=False)
    print("done")


if __name__ == "__main__":
    sizes = [0.8, 0.20]  # training, testing sets ratios
    y_field = "LOG_SITE_ENERGY_kWh_yr"
    data_energy_folder = DATA_RAW_BUILDING_PERFORMANCE_FOLDER
    data_ipcc_file = DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE
    outputpath_training = DATA_TRAINING_FILE
    outputpath_testing = DATA_TESTING_FILE
    outputpath_all_data = DATA_ALLDATA_FILE
    scenarios = "data_1990_2010"

    data_efficiency = pd.read_excel(DATA_FUTURE_EFFICIENCY_FILE, sheet_name="data").set_index('year')
    today_efficiency = data_efficiency.loc[2010]
    temperatures_base_H_C = today_efficiency["Tb_H_A1B"]
    temperatures_base_C_C = today_efficiency["Tb_H_A1B"]
    relative_humidity_base_HUM_C = today_efficiency["RH_HUM_A1B"]
    relative_humidity_base_DEHUM_C = today_efficiency["RH_DEHUM_A1B"]
    COP_H = today_efficiency["COP_H_A1B"]
    COP_C = today_efficiency["COP_C_A1B"]

    cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')['City'].values
    main(cities, data_energy_folder, data_ipcc_file, outputpath_training, outputpath_testing, outputpath_all_data,
         y_field, sizes, scenarios, COP_H, COP_C, temperatures_base_H_C, temperatures_base_C_C,
         relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C)
