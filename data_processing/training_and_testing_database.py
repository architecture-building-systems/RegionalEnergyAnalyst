import os

import numpy as np
import pandas as pd
from sklearn import mixture
from sklearn.model_selection import train_test_split

from configuration import DATA_RAW_BUILDING_PERFORMANCE_FOLDER, CONFIG_FILE, DATA_TRAINING_FILE, \
    DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER, DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE, DATA_TESTING_FILE, \
    DATA_ALLDATA_FILE, DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE, DATA_RAW_BUILDING_TODAY_HDD_FOLDER, \
    DATA_FUTURE_EFFICIENCY_FILE, ZONE_NAMES
from data_processing.enthalpy_calculation import convert_rh_to_moisture_content, calc_yearly_enthalpy


def prepare_input_database(cities, data_energy_folder, data_ipcc_folder, scenario, COP_H, COP_C,
                           temperatures_base_H_C, temperatures_base_C_C,
                           relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C, climate):
    final_df = pd.DataFrame()
    name_of_data_file = [x.split(",")[0] + "_" + x.split(", ")[-1] + "-hour.dat" for x in cities]
    name_of_data_file = [x.replace(" ", "_") for x in name_of_data_file]
    data_1990_2010 = pd.read_csv(DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE)

    # get climate calssification
    new_clima = []
    for clima in climate:
        for category, categories in ZONE_NAMES.items():
            if clima.split(" ")[0] in categories:
                new_clima.append(category)

    for name_file, city, climate in zip(name_of_data_file, cities, new_clima):
        print(city)
        # get_local_data:
        data_measured = pd.read_csv(os.path.join(data_energy_folder, city + ".csv"))
        data_measured["site_year"] = data_measured["site_year"].round(0)
        df_hdd_cdd = pd.read_csv(os.path.join(DATA_RAW_BUILDING_TODAY_HDD_FOLDER, city + ".csv"))
        df_hdd_cdd["site_year"] = df_hdd_cdd["site_year"].round(0)
        data = pd.merge(df_hdd_cdd, data_measured, on="site_year")
        data["BUILDING_ID"] = [city + str(ix) for ix in data.index]

        # get enthalpy
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

        data['CITY'] = city
        data['CLIMATE_ZONE'] = climate
        data['SCENARIO'] = scenario


        # Standarize site energy consumption
        hhd_standard_year = data_1990_2010[data_1990_2010["City"] == city]
        hhd_standard_year = hhd_standard_year[hhd_standard_year["Scenario"] == "1990_2010"]
        hdd_cdd_standard = hhd_standard_year["HDD_18_5_C"].values[0] + hhd_standard_year["CDD_18_5_C"].values[0]
        data["SITE_ENERGY_kWh_yr"] = ((data["site_energy"] * 0.293071).round(2) *
                                      (data["hdd_18.5C"] + data["cdd_18.5C"])) / hdd_cdd_standard
        data["SITE_EUI_kWh_m2yr"] = ((data["site_eui"] * 3.15459).round(2) *
                                     (data["hdd_18.5C"] + data["cdd_18.5C"])) / hdd_cdd_standard


        # final calculation
        data["GROSS_FLOOR_AREA_m2"] = (data["floor_area"] * 0.092903).values
        data["BUILDING_CLASS"] = data["building_class"].values
        volumetric_flow_building_m3_s = np.vectorize(calc_massflow_building)(data["GROSS_FLOOR_AREA_m2"].values,
                                                                             data["BUILDING_CLASS"].values)
        data["THERMAL_ENERGY_kWh_yr"] = [calc_total_energy(x, delta_enthalpy_kJ_kg, y, COP_H, COP_C) for x, y in
                                         zip(volumetric_flow_building_m3_s, data["BUILDING_CLASS"].values)]

        #logarithmic values
        data['LOG_THERMAL_ENERGY_kWh_yr'] = np.log(data["THERMAL_ENERGY_kWh_yr"].values)
        data['LOG_SITE_EUI_kWh_m2yr'] = np.log(data["SITE_EUI_kWh_m2yr"].values)
        data['LOG_SITE_ENERGY_kWh_yr'] = np.log(data["SITE_ENERGY_kWh_yr"].values)

        data = calc_clusters(data)

        # list of fields to extract
        fields = ["BUILDING_ID",
                  "CITY",
                  "CLIMATE_ZONE",
                  "SCENARIO",
                  "BUILDING_CLASS",
                  "GROSS_FLOOR_AREA_m2",
                  "LOG_SITE_EUI_kWh_m2yr",
                  "LOG_SITE_ENERGY_kWh_yr",
                  "LOG_THERMAL_ENERGY_kWh_yr",
                  "CLUSTER_LOG_SITE_EUI_kWh_m2yr"
                  ]
        final_df = pd.concat([final_df, data[fields]], ignore_index=True)
        print("scenario and city done: ", scenario, city)
    return final_df


def calc_clusters(data):
    random_state = 170
    n_components = 5
    building_classes = data.BUILDING_CLASS.unique()
    df = pd.DataFrame()
    for j, building_class in enumerate(building_classes):
        df3 = data[data["BUILDING_CLASS"] == building_class]
        if df3.empty:
            x = 1
        else:
            X_cluster = df3[["LOG_SITE_EUI_kWh_m2yr"]].values
            cv_type = 'tied'
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=random_state)
            gmm.fit(X_cluster)
            means = gmm.means_.T[0]  # /gmm.means_.T[1]
            cluster_labels = gmm.predict(X_cluster)
            df3['CLUSTER_LOG_SITE_EUI_kWh_m2yr'] = [round(means[cluster], 2) for cluster in cluster_labels]
            df = pd.concat([df, df3], ignore_index=True)

    df = data.merge(df[['BUILDING_ID','CLUSTER_LOG_SITE_EUI_kWh_m2yr']], on='BUILDING_ID')

    return df


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
        raise Exception("errorrrr")

    volumetric_flow_building_m3_s = ACH_1_h * gross_floor_area_m2 * F / 3600

    return volumetric_flow_building_m3_s


def training_testing_splitter(final_df, cities, y_field, sizes):
    random_state = 170
    test_size = sizes[1]
    X_final_train = pd.DataFrame()
    X_final_test = pd.DataFrame()
    # split into training and test datasets
    building_class = ["Residential", "Commercial"]
    for city in cities:
        data = final_df[final_df['CITY'] == city]
        for bu_class in building_class:
            data_final = data[data['BUILDING_CLASS'] == bu_class]
            if data_final.empty:
                x=1
            else:
                y = data_final[y_field]
                X_train, X_test, y_train, y_test = train_test_split(data_final, y, test_size=test_size,
                                                                    random_state=random_state)

                # join each dataset
                X_final_train = pd.concat([X_final_train, X_train], ignore_index=True)
                X_final_test = pd.concat([X_final_test, X_test], ignore_index=True)

    return X_final_train, X_final_test


def main(cities, data_energy_folder, data_ipcc_folder, outputpath_training, outputpath_testing, outputpath_all_data,
         y_field, sizes, scenarios, COP_H, COP_C, temperatures_base_H_C, temperatures_base_C_C,
         relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C, climate):

    final_df = prepare_input_database(cities, data_energy_folder, data_ipcc_folder, scenarios, COP_H, COP_C,
                                      temperatures_base_H_C, temperatures_base_C_C,
                                      relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C, climate)

    training, testing = training_testing_splitter(final_df, cities, y_field, sizes)

    training.to_csv(outputpath_training, index=False)
    testing.to_csv(outputpath_testing, index=False)
    final_df.to_csv(outputpath_all_data, index=False)
    print("done")


if __name__ == "__main__":
    sizes = [0.8, 0.20]  # training, testing sets ratios
    y_field = "LOG_SITE_ENERGY_kWh_yr"
    data_energy_folder = DATA_RAW_BUILDING_PERFORMANCE_FOLDER
    data_ipcc_folder = DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER
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
    climate = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')['climate'].values
    # cities = ['Cheyenne, WY']
    main(cities, data_energy_folder, data_ipcc_folder, outputpath_training, outputpath_testing, outputpath_all_data,
         y_field, sizes, scenarios, COP_H, COP_C, temperatures_base_H_C, temperatures_base_C_C,
         relative_humidity_base_HUM_C, relative_humidity_base_DEHUM_C, climate)
